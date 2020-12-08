// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_
#define TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/batch_matmul_op_impl.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace custom_helper_op {
static const int kMaxParallelImgs = 32;

struct SparseConv3DFastParams {
  int32 input_batches;
  int32 input_channels;
  int32 input_rows;
  int32 input_cols;
  int32 input_depths;
  int32 filter_channels;
  int32 filter_rows;
  int32 filter_cols;
  int32 filter_depths;
  int32 padding_rows;
  int32 padding_cols;
  int32 padding_depths;
  int32 stride_rows;
  int32 stride_cols;
  int32 stride_depths;
  int32 dilation_rows;
  int32 dilation_cols;
  int32 dilation_depths;
  int32 output_channels;
  int32 output_rows;
  int32 output_cols;
  int32 output_depths;
  int32 parallel_imgs;
  int32 batches;
  bool dynamic_default;
};

template <typename Device, typename T>
Status TensorSetZero(OpKernelContext *ctx, Tensor *value) {
  const auto d = ctx->template eigen_device<Device>();
  auto out = value->flat<T>();

  const bool use_64bit = out.size() > Eigen::NumTraits<int>::highest();

  if (!use_64bit && Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    To32Bit(out).device(d) = To32Bit(out).constant(T(0));
  } else {
    out.device(d) = out.constant(T(0));
  }

  return Status::OK();
}

template <typename Device, typename T>
Status AddToTensor(OpKernelContext *ctx, Tensor *sum, const Tensor *current,
                   const Tensor *add) {
  const auto d = ctx->template eigen_device<Device>();

  auto out = sum->flat<T>();
  auto a = current->flat<T>();
  auto b = add->flat<T>();

  const bool use_64bit = out.size() > Eigen::NumTraits<int>::highest();

  if (!use_64bit && Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    To32Bit(out).device(d) = To32Bit(a) + To32Bit(b);
  } else {
    out.device(d) = a + b;
  }

  return Status::OK();
}

template <typename Device, typename T, int NDIMS>
Status Transpose(OpKernelContext *ctx, const Tensor &in,
                 const gtl::ArraySlice<int32> perm, Tensor *out) {
  const auto d = ctx->template eigen_device<Device>();

  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) {
    p[i] = perm[i];
  }

  auto x = typename TTypes<T, NDIMS>::ConstTensor(
      reinterpret_cast<const T *>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<NDIMS>());
  auto y = typename TTypes<T, NDIMS>::Tensor(
      reinterpret_cast<T *>(const_cast<char *>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<NDIMS>());

  const bool use_64bit = x.size() > Eigen::NumTraits<int>::highest();

  if (!use_64bit && Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    To32Bit(y).device(d) = To32Bit(x).shuffle(p);
  } else {
    y.device(d) = x.shuffle(p);
  }

  return Status::OK();
}

template <typename Device, typename T>
Status CopySliceToElement(OpKernelContext *ctx, const Tensor &parent,
                          Tensor *element, int64 index) {
  const auto d = ctx->template eigen_device<Device>();

  auto out = element->flat<T>();
  auto in = parent.flat_outer_dims<T>();

  const bool use_64bit = in.size() > Eigen::NumTraits<int>::highest();

  if (!use_64bit && Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    To32Bit(out).device(d) = To32Bit(in).chip(index, 0);
  } else {
    out.device(d) = in.chip(index, 0);
  }

  return Status::OK();
}

template <typename Device, typename T>
Status CopyElementToSlice(OpKernelContext *ctx, Tensor element, Tensor *parent,
                          int64 index) {
  const auto d = ctx->template eigen_device<Device>();

  auto out = parent->flat_outer_dims<T>();
  auto in = element.flat<T>();

  const bool use_64bit = out.size() > Eigen::NumTraits<int>::highest();

  if (!use_64bit && Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    To32Bit(out).chip(index, 0).device(d) = To32Bit(in);
  } else {
    out.chip(index, 0).device(d) = in;
  }

  return Status::OK();
}

namespace functor {

template <typename T>
EIGEN_DEVICE_FUNC T BilinearInterpolate(typename TTypes<T, 5>::Tensor img,
                                        int32 b, int32 batch, int32 channel,
                                        T y, T x) {
  const auto max_height = img.dimension(3);
  const auto max_width = img.dimension(4);

  if (y <= -1 || max_height <= y || x <= -1 || max_width <= x) {
    return T(0);
  }

  int y_low = floor(y);
  int x_low = floor(x);
  int y_high = y_low + 1;
  int w_high = x_low + 1;

  auto v1 = T(0);
  if (y_low >= 0 && x_low >= 0) {
    v1 = img(b, batch, channel, y_low, x_low);
  }

  auto v2 = T(0);
  if (y_low >= 0 && w_high <= max_width - 1) {
    v2 = img(b, batch, channel, y_low, w_high);
  }

  auto v3 = T(0);
  if (y_high <= max_height - 1 && x_low >= 0) {
    v3 = img(b, batch, channel, y_high, x_low);
  }

  auto v4 = T(0);
  if (y_high <= max_height - 1 && w_high <= max_width - 1) {
    v4 = img(b, batch, channel, y_high, w_high);
  }

  auto lh = y - y_low;
  auto lw = x - x_low;
  auto hh = 1 - lh;
  auto hw = 1 - lw;

  auto w1 = hh * hw;
  auto w2 = hh * lw;
  auto w3 = lh * hw;
  auto w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename T>
EIGEN_DEVICE_FUNC T GetCoordinateWeight(typename TTypes<T, 5>::Tensor img,
                                        int32 b, int32 batch, int32 channel,
                                        T y, T x, bool is_y_direction) {
  const auto max_height = img.dimension(3);
  const auto max_width = img.dimension(4);

  const int y_low = floor(y);
  const int x_low = floor(x);
  const int y_high = y_low + 1;
  const int x_high = x_low + 1;

  const bool valid_y_low = max_height > y_low && y_low >= 0;
  const bool valid_y_high = max_height > y_high && y_high >= 0;
  const bool valid_x_low = max_width > x_low && x_low >= 0;
  const bool valid_x_high = max_width > x_high && x_high >= 0;

  auto v_yx = T(0);
  if (valid_y_low && valid_x_low) {
    v_yx = img(b, batch, channel, y_low, x_low);
  }

  auto v_yX = T(0);
  if (valid_y_low && valid_x_high) {
    v_yX = img(b, batch, channel, y_low, x_high);
  }

  auto v_Yx = T(0);
  if (valid_y_high && valid_x_low) {
    v_Yx = img(b, batch, channel, y_high, x_low);
  }

  auto v_YX = T(0);
  if (valid_y_high && valid_x_high) {
    v_YX = img(b, batch, channel, y_high, x_high);
  }

  if (is_y_direction) {
    const auto dx = x - x_low;
    return (v_YX - v_yX) * dx + (v_Yx - v_yx) * (1 - dx);
  } else {
    const auto dy = y - y_low;
    return (v_YX - v_Yx) * dy + (v_yX - v_yx) * (1 - dy);
  }
}

template <typename Device, typename T>
struct SparseConv3DFastFunctorBase {
  SparseConv3DFastFunctorBase(const Tensor *_input_tensor,
                              const Tensor *_filter_tensor,
                              const Tensor *_bias_tensor,
                              const Tensor *_base_plane_tensor,
                              const Tensor *_mask_tensor,
                              Tensor *_column_buffer_tensor,
                              SparseConv3DFastParams *_p)
      : input_tensor(_input_tensor->dtype()),
        filter_tensor(_filter_tensor->dtype()),
        bias_tensor(_bias_tensor->dtype()),
        base_plane_tensor(_base_plane_tensor->dtype()),
        column_buffer_tensor(_column_buffer_tensor->dtype()),
        p(*_p) {

  }

  virtual Status operator()(OpKernelContext *context) = 0;

  void DeformableIm2Col(OpKernelContext *context, int32 b);

  Tensor input_tensor;
  Tensor filter_tensor;
  Tensor bias_tensor;
  Tensor base_plane_tensor;
  Tensor column_buffer_tensor;
  SparseConv3DFastParams p;
};

template <typename Device, typename T>
struct SparseConv3DFastForwardFunctor
    : public SparseConv3DFastFunctorBase<Device, T> {
  using SparseConv3DFastFunctorBase<Device, T>::input_tensor;
  using SparseConv3DFastFunctorBase<Device, T>::filter_tensor;
  using SparseConv3DFastFunctorBase<Device, T>::bias_tensor;
  using SparseConv3DFastFunctorBase<Device, T>::base_plane_tensor;
  using SparseConv3DFastFunctorBase<Device, T>::column_buffer_tensor;
  using SparseConv3DFastFunctorBase<Device, T>::p;

  SparseConv3DFastForwardFunctor(
      const Tensor *_input_tensor, const Tensor *_filter_tensor,
      const Tensor *_bias_tensor, const Tensor *_offset_tensor,
      const Tensor *_mask_tensor, Tensor *_column_buffer_tensor,
      Tensor *_output_tensor, SparseConv3DFastParams *_p)
      : SparseConv3DFastFunctorBase<Device, T>(
            _input_tensor, _filter_tensor, _bias_tensor, _offset_tensor,
            _mask_tensor, _column_buffer_tensor, _p),
        output_tensor(_output_tensor->dtype()) {
    CHECK(output_tensor.CopyFrom(*_output_tensor, _output_tensor->shape()));
  }

  Status operator()(OpKernelContext *context) {

    return Status::OK();
  }

  Tensor output_tensor;
};

template <typename Device, typename T>
struct SparseConv3DFastGradFunctor
    : public SparseConv3DFastFunctorBase<Device, T> {
  using SparseConv3DFastFunctorBase<Device, T>::input_tensor;
  using SparseConv3DFastFunctorBase<Device, T>::filter_tensor;
  using SparseConv3DFastFunctorBase<Device, T>::bias_tensor;
  using SparseConv3DFastFunctorBase<Device, T>::base_plane_tensor;
  using SparseConv3DFastFunctorBase<Device, T>::column_buffer_tensor;
  using SparseConv3DFastFunctorBase<Device, T>::p;

  SparseConv3DFastGradFunctor(
      const Tensor *_input_tensor, const Tensor *_filter_tensor,
      const Tensor *_bias_tensor, const Tensor *_offset_tensor,
      const Tensor *_mask_tensor, Tensor *_output_grad_tensor,
      Tensor *_input_grad_tensor, Tensor *_filter_grad_tensor,
      Tensor *_bias_grad_tensor, Tensor *_offset_grad_tensor,
      Tensor *_mask_grad_tensor, Tensor *_column_buffer_tensor,
      SparseConv3DFastParams *_p)
      : SparseConv3DFastFunctorBase<Device, T>(
            _input_tensor, _filter_tensor, _bias_tensor, _offset_tensor,
            _mask_tensor, _column_buffer_tensor, _p),
        output_grad_tensor(_output_grad_tensor->dtype()),
        input_grad_tensor(_input_grad_tensor->dtype()),
        filter_grad_tensor(_filter_grad_tensor->dtype()),
        bias_grad_tensor(_bias_grad_tensor->dtype()),
        offset_grad_tensor(_offset_grad_tensor->dtype()),
        mask_grad_tensor(_mask_grad_tensor->dtype()) {

  }

  Status operator()(OpKernelContext *context) {

    return Status::OK();
  }

  void ComputeFilterGrad(OpKernelContext *context) {

  }

  void ComputeInputOffsetMaskGrad(OpKernelContext *context) {

  }

  void DeformableCol2ImForOffsetAndMask(OpKernelContext *context, int32 b);

  void DeformableCol2ImForInput(OpKernelContext *context, int32 b);

  Tensor output_grad_tensor;
  Tensor input_grad_tensor;
  Tensor filter_grad_tensor;
  Tensor bias_grad_tensor;
  Tensor offset_grad_tensor;
  Tensor mask_grad_tensor;
};

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_