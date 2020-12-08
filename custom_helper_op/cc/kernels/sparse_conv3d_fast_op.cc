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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "custom_helper_op/cc/kernels/sparse_conv3d_fast_op.h"

#include <array>
#include <mutex>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/kernel_shape_util.h"

namespace tensorflow {
namespace custom_helper_op {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#if GOOGLE_CUDA
#define EXTERN_TEMPLATE(T)                           \
  extern template Status Transpose<GPUDevice, T, 5>( \
      OpKernelContext * ctx, const Tensor &in,       \
      const gtl::ArraySlice<int32> perm, Tensor *out);
TF_CALL_float(EXTERN_TEMPLATE);
TF_CALL_double(EXTERN_TEMPLATE);
#undef EXTERN_TEMPLATE
#endif  // GOOGLE_CUDA

namespace functor {

#if GOOGLE_CUDA
#define EXTERN_TEMPLATE(T)                                             \
  extern template struct SparseConv3DFastForwardFunctor<GPUDevice, T>; \
  extern template struct SparseConv3DFastGradFunctor<GPUDevice, T>;
TF_CALL_float(EXTERN_TEMPLATE);
TF_CALL_double(EXTERN_TEMPLATE);
#undef EXTERN_TEMPLATE
#endif  // GOOGLE_CUDA

#define IM2COL(T)                                                              \
  template <>                                                                  \
  void SparseConv3DFastFunctorBase<CPUDevice, T>::DeformableIm2Col(            \
      OpKernelContext *context, int32 b) {                                     \
  }
TF_CALL_float(IM2COL);
TF_CALL_double(IM2COL);
#undef IM2COL

#define COL2IM_OFFSET_AND_MASK(T)                                              \
  template <>                                                                  \
  void                                                                         \
  SparseConv3DFastGradFunctor<CPUDevice, T>::DeformableCol2ImForOffsetAndMask( \
      OpKernelContext *context, int32 b) {                                     \
  }
TF_CALL_float(COL2IM_OFFSET_AND_MASK);
TF_CALL_double(COL2IM_OFFSET_AND_MASK);
#undef COL2IM_OFFSET_AND_MASK

#define COL2IM_INPUT(T)                                                        \
  template <>                                                                  \
  void SparseConv3DFastGradFunctor<CPUDevice, T>::DeformableCol2ImForInput(    \
      OpKernelContext *context, int32 b) {                                     \
  }
TF_CALL_float(COL2IM_INPUT);
TF_CALL_double(COL2IM_INPUT);
#undef COL2IM_INPUT

}  // end namespace functor

template <typename Device, typename T>
class SparseConv3DFastOpBase : public OpKernel {
 public:
  explicit SparseConv3DFastOpBase(OpKernelConstruction *context)
      : OpKernel(context), p{} {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    FormatFromString(data_format_str, &data_format);

    OP_REQUIRES_OK(context, context->GetAttr("dynamic_default", &dynamic_default));
    OP_REQUIRES(context, strides.size() == 3, errors::InvalidArgument("strides must be a vector have 3 element"));
    OP_REQUIRES(context, dilations.size() == 3, errors::InvalidArgument("dilations must be a vector have 3 element"));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor& images = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& default_channels_value = context->input(2);
    const Tensor& base_plane = context->input(3);


    OP_REQUIRES(context, TensorShapeUtils::IsScalar(default_channels_value.shape()),
                errors::InvalidArgument("default_value must be scalar: ",
                                        default_channels_value.shape().DebugString()));
    OP_REQUIRES(context, images.shape().dims() == 5,
                errors::InvalidArgument("images must have rank 4"));

    const auto input_batches = images.dim_size(0);
    const auto input_rows = images.dim_size(1);
    const auto input_cols = images.dim_size(2);
    const auto input_depths = images.dim_size(3);
    const auto input_channels = images.dim_size(4);

    OP_REQUIRES(context, (filter.shape().dims() == 5) && (filter.dim_size(3) == input_channels),
                errors::InvalidArgument("filter must have rank 5, and must compate to ref_image"));
    const auto filter_rows = filter.dim_size(0);
    const auto filter_cols = filter.dim_size(1);
    const auto filter_depths = filter.dim_size(2);
    const auto output_channels = filter.dim_size(4);

    OP_REQUIRES(context, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == input_batches)
                      && (base_plane.dim_size(1) == input_rows) && (base_plane.dim_size(2) == input_cols) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to ref_image, got ", base_plane.shape().DebugString()));

    int64 output_rows, output_cols, output_depths;
    int64 padding_rows, padding_cols, padding_depths;
    OP_REQUIRES_OK(
        context, GetWindowedOutputSizeV2(input_rows, filter_rows, dilations[0],
                                         strides[0], Padding::SAME, &output_rows,
                                         &padding_rows));
    OP_REQUIRES_OK(
        context, GetWindowedOutputSizeV2(input_cols, filter_cols, dilations[1],
                                         strides[1], Padding::SAME, &output_cols,
                                         &padding_cols));

    OP_REQUIRES_OK(
        context, GetWindowedOutputSizeV2(input_cols, filter_cols, dilations[2],
                                         strides[2], Padding::SAME, &output_depths,
                                         &padding_depths));

    const auto parallel_imgs = GetParallelImgs(input_batches);

    p.input_batches = input_batches;
    p.input_channels = input_channels;
    p.input_rows = input_rows;
    p.input_cols = input_cols;
    p.input_depths = input_depths;
    p.filter_channels = input_channels;
    p.filter_rows = filter_rows;
    p.filter_cols = filter_cols;
    p.filter_depths = filter_depths;
    p.padding_rows = padding_rows;
    p.padding_cols = padding_cols;
    p.padding_depths = padding_depths;
    p.stride_rows = strides[0];
    p.stride_cols = strides[1];
    p.stride_depths = strides[2];
    p.dilation_rows = dilations[0];
    p.dilation_cols = dilations[1];
    p.dilation_depths = dilations[2];
    p.output_channels = output_channels;
    p.output_rows = output_rows;
    p.output_cols = output_cols;
    p.output_depths = output_depths;
    p.parallel_imgs = parallel_imgs;
    p.batches = p.input_batches / p.parallel_imgs;
    p.dynamic_default = dynamic_default;
  }

  int GetParallelImgs(int n) {
    for (auto k = kMaxParallelImgs; k > 1; --k) {
      if (n % k == 0) {
        return k;
      }
    }
    return 1;
  }

 protected:
  TensorFormat data_format;
  SparseConv3DFastParams p;
  bool dynamic_default;

 private:
  std::vector<int32> strides;
  std::vector<int32> dilations;
};

template <typename Device, typename T>
class SparseConv3DFastForwardOp : public SparseConv3DFastOpBase<Device, T> {
  using SparseConv3DFastOpBase<Device, T>::data_format;
  using SparseConv3DFastOpBase<Device, T>::p;

 public:
  explicit SparseConv3DFastForwardOp(OpKernelConstruction *context)
      : SparseConv3DFastOpBase<Device, T>(context) {}

  void Compute(OpKernelContext *context) override {
    SparseConv3DFastOpBase<Device, T>::Compute(context);

    const Tensor &input_tensor = context->input(0);
    const Tensor &filter_tensor = context->input(1);
    const Tensor &bias_tensor = context->input(2);
    const Tensor &offset_tensor = context->input(3);
    const Tensor &mask_tensor = context->input(4);

    TensorShape column_buffer_shape(
        {p.input_channels * p.filter_rows * p.filter_cols, p.parallel_imgs,
         p.output_rows, p.output_cols});
    Tensor column_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   column_buffer_shape,
                                                   &column_buffer_tensor));

    TensorShape output_shape =
        ShapeFromFormat(data_format, p.input_batches, p.output_rows,
                        p.output_cols, p.output_channels);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    functor::SparseConv3DFastForwardFunctor<Device, T> deformableConv2DFunc(
        &input_tensor, &filter_tensor, &bias_tensor, &offset_tensor,
        &mask_tensor, &column_buffer_tensor, output_tensor, &p);
    Status s = deformableConv2DFunc(context);

    OP_REQUIRES_OK(context, s);
  }
};

template <typename Device, typename T>
class SparseConv3DFastGradOp : public SparseConv3DFastOpBase<Device, T> {
  using SparseConv3DFastOpBase<Device, T>::data_format;
  using SparseConv3DFastOpBase<Device, T>::p;

 public:
  explicit SparseConv3DFastGradOp(OpKernelConstruction *context)
      : SparseConv3DFastOpBase<Device, T>(context) {}

  void Compute(OpKernelContext *context) override {
    SparseConv3DFastOpBase<Device, T>::Compute(context);

    const Tensor &input_tensor = context->input(0);
    const Tensor &filter_tensor = context->input(1);
    const Tensor &bias_tensor = context->input(2);
    const Tensor &offset_tensor = context->input(3);
    const Tensor &mask_tensor = context->input(4);
    const Tensor &output_grad_tensor = context->input(5);

    const TensorShape &input_shape = input_tensor.shape();
    const TensorShape &filter_shape = filter_tensor.shape();
    const TensorShape &bias_shape = bias_tensor.shape();
    const TensorShape &offset_shape = offset_tensor.shape();
    const TensorShape &mask_shape = mask_tensor.shape();

    TensorShape column_buffer_shape(
        {p.input_channels * p.filter_rows * p.filter_cols, p.parallel_imgs,
         p.output_rows, p.output_cols});
    Tensor column_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   column_buffer_shape,
                                                   &column_buffer_tensor));

    Tensor output_grad_tensor_reshaped;
    CHECK(output_grad_tensor_reshaped.CopyFrom(
        output_grad_tensor,
        TensorShape({p.batches, p.parallel_imgs, p.output_channels,
                     p.output_rows, p.output_cols})));

    TensorShape output_grad_tensor_transposed_shape(
        {p.batches, p.output_channels, p.parallel_imgs, p.output_rows,
         p.output_cols});
    Tensor output_grad_tensor_transposed;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          output_grad_tensor_transposed_shape,
                                          &output_grad_tensor_transposed));
    OP_REQUIRES_OK(context,
                   Transpose<Device, T, 5>(context, output_grad_tensor_reshaped,
                                           {0, 2, 1, 3, 4},
                                           &output_grad_tensor_transposed));

    TensorShape output_shape =
        ShapeFromFormat(data_format, p.input_batches, p.output_rows,
                        p.output_cols, p.output_channels);

    Tensor *input_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, input_shape, &input_grad_tensor));
    Tensor *filter_grad_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, filter_shape,
                                                     &filter_grad_tensor));
    Tensor *bias_grad_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, bias_shape, &bias_grad_tensor));
    Tensor *offset_grad_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, offset_shape,
                                                     &offset_grad_tensor));
    Tensor *mask_grad_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, mask_shape, &mask_grad_tensor));

    functor::SparseConv3DFastGradFunctor<Device, T> deformableConv2DGradFunc(
        &input_tensor, &filter_tensor, &bias_tensor, &offset_tensor,
        &mask_tensor, &output_grad_tensor_transposed, input_grad_tensor,
        filter_grad_tensor, bias_grad_tensor, offset_grad_tensor,
        mask_grad_tensor, &column_buffer_tensor, &p);
    Status s = deformableConv2DGradFunc(context);

    OP_REQUIRES_OK(context, s);
  }
};

// Register the CPU kernels.
#define REGISTER_SPARSE_CONV3D_FAST_OP_CPU(T)                        \
  REGISTER_KERNEL_BUILDER(Name("SparseConv3DFast")          \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<T>("dtype"),             \
                          SparseConv3DFastForwardOp<CPUDevice, T>) \
  REGISTER_KERNEL_BUILDER(Name("SparseConv3DFastGrad")      \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<T>("dtype"),             \
                          SparseConv3DFastGradOp<CPUDevice, T>)

TF_CALL_float(REGISTER_SPARSE_CONV3D_FAST_OP_CPU);
TF_CALL_double(REGISTER_SPARSE_CONV3D_FAST_OP_CPU);
#undef REGISTER_DEFORMABLECONV2D_OP_CPU

// Register the GPU kernels.
#if GOOGLE_CUDA

#define REGISTER_DEFORMABLECONV2D_OP_GPU(T)                        \
  REGISTER_KERNEL_BUILDER(Name("SparseConv3DFast")          \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("dtype"),             \
                          SparseConv3DFastForwardOp<GPUDevice, T>) \
  REGISTER_KERNEL_BUILDER(Name("SparseConv3DFastGrad")      \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("dtype"),             \
                          SparseConv3DFastGradOp<GPUDevice, T>)

TF_CALL_float(REGISTER_DEFORMABLECONV2D_OP_GPU);
TF_CALL_double(REGISTER_DEFORMABLECONV2D_OP_GPU);
#undef REGISTER_DEFORMABLECONV2D_OP_GPU

#endif  // GOOGLE_CUDA

}  // namespace addons
}  // namespace tensorflow