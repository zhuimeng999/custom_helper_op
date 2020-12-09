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

namespace functor {

template<typename Device, typename T>
void SparseConv3DFastFunctor<Device, T>::operator()(const Device& d, const SparseConv3DFastParams &p)
{

};

template struct SparseConv3DFastFunctor<CPUDevice, float>;
template struct SparseConv3DFastFunctor<CPUDevice, double>;

}  // end namespace functor

using functor::SparseConv3DFastFunctor;
using functor::SparseConv3DFastGradFunctor;

template <typename Device, typename T>
class SparseConv3DFastOpBase : public OpKernel {
 public:
  explicit SparseConv3DFastOpBase(OpKernelConstruction *ctx)
      : OpKernel(ctx), p{} {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations));
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    FormatFromString(data_format_str, &data_format);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_default", &dynamic_default));
    OP_REQUIRES(ctx, strides.size() == 3, errors::InvalidArgument("strides must be a vector have 3 element"));
    OP_REQUIRES(ctx, dilations.size() == 3, errors::InvalidArgument("dilations must be a vector have 3 element"));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor& images = ctx->input(0);
    const Tensor& filter = ctx->input(1);
    const Tensor& default_channels_value = ctx->input(2);
    const Tensor& base_plane = ctx->input(3);


    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(default_channels_value.shape()),
                errors::InvalidArgument("default_value must be scalar: ",
                                        default_channels_value.shape().DebugString()));
    OP_REQUIRES(ctx, images.shape().dims() == 5,
                errors::InvalidArgument("images must have rank 4"));

    const auto input_batches = images.dim_size(0);
    const auto input_rows = images.dim_size(1);
    const auto input_cols = images.dim_size(2);
    const auto input_depths = images.dim_size(3);
    const auto input_channels = images.dim_size(4);

    OP_REQUIRES(ctx, (filter.shape().dims() == 5) && (filter.dim_size(3) == input_channels),
                errors::InvalidArgument("filter must have rank 5, and must compate to ref_image"));
    const auto filter_rows = filter.dim_size(0);
    const auto filter_cols = filter.dim_size(1);
    const auto filter_depths = filter.dim_size(2);
    const auto output_channels = filter.dim_size(4);

    OP_REQUIRES(ctx, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == input_batches)
                      && (base_plane.dim_size(1) == input_rows) && (base_plane.dim_size(2) == input_cols) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to ref_image, got ", base_plane.shape().DebugString()));

    int64 output_rows, output_cols, output_depths;
    int64 padding_rows, padding_cols, padding_depths;
    OP_REQUIRES_OK(
        ctx, GetWindowedOutputSizeV2(input_rows, filter_rows, dilations[0],
                                         strides[0], Padding::SAME, &output_rows,
                                         &padding_rows));
    OP_REQUIRES_OK(
        ctx, GetWindowedOutputSizeV2(input_cols, filter_cols, dilations[1],
                                         strides[1], Padding::SAME, &output_cols,
                                         &padding_cols));

    OP_REQUIRES_OK(
        ctx, GetWindowedOutputSizeV2(input_cols, filter_cols, dilations[2],
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

using functor::SparseConv3DFastFunctor;

template <typename Device, typename T>
class SparseConv3DFastOp : public SparseConv3DFastOpBase<Device, T> {
  using SparseConv3DFastOpBase<Device, T>::data_format;
  using SparseConv3DFastOpBase<Device, T>::p;

 public:
  explicit SparseConv3DFastOp(OpKernelConstruction* ctx)
      : SparseConv3DFastOpBase<Device, T>(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    SparseConv3DFastOpBase<Device, T>::Compute(ctx);

    const Tensor &input_tensor = ctx->input(0);
    const Tensor &filter_tensor = ctx->input(1);
    const Tensor &default_value_tensor = ctx->input(2);
    const Tensor &base_plane_tensor = ctx->input(3);

    TensorShape column_buffer_shape(
        {p.input_channels * p.filter_rows * p.filter_cols * p.filter_depths, p.parallel_imgs,
         p.output_rows, p.output_cols, p.output_depths});
    Tensor column_buffer_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                                   column_buffer_shape,
                                                   &column_buffer_tensor));

    TensorShape output_shape =
        ShapeFromFormat(data_format, p.input_batches, {p.output_rows,
                        p.output_cols, p.output_depths}, p.output_channels);
    std::cout << output_shape.DebugString();
    // VLOG(WARNING) << "######################" << output_shape;
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, output_shape, &output_tensor));

    SparseConv3DFastFunctor<Device, T>()(ctx->eigen_device<Device>(), p);
  }
private:
  TF_DISALLOW_COPY_AND_ASSIGN(SparseConv3DFastOp);
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("SparseConv3DFast") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          SparseConv3DFastOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
// TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

using functor::SparseConv3DFastGradFunctor;

template <typename Device, typename T>
class SparseConv3DFastGradOp : public OpKernel {
 private:
  std::vector<int> strides_;
  std::vector<int> dilations_;
  bool dynamic_default_;
 public:
  explicit SparseConv3DFastGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_default", &dynamic_default_));
    OP_REQUIRES(ctx, strides_.size() == 3, errors::InvalidArgument("strides must be a vector have 3 element"));
    OP_REQUIRES(ctx, dilations_.size() == 3, errors::InvalidArgument("dilations must be a vector have 3 element"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& images = ctx->input(0);
    const Tensor& filter = ctx->input(1);
    const Tensor& default_channels_value = ctx->input(2);
    const Tensor& base_plane = ctx->input(3);

    const Tensor& out_grad = ctx->input(4);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(default_channels_value.shape()),
                errors::InvalidArgument("default_value must be scalar: ",
                                        default_channels_value.shape().DebugString()));
    OP_REQUIRES(ctx, images.shape().dims() == 5,
                errors::InvalidArgument("images must have rank 5"));

    const auto batch_size = images.dim_size(0);
    const auto image_height = images.dim_size(1);
    const auto image_width = images.dim_size(2);
    const auto image_depth = images.dim_size(3);
    const auto image_channels = images.dim_size(4);

    OP_REQUIRES(ctx, (filter.shape().dims() == 5) && (filter.dim_size(3) == image_channels),
                errors::InvalidArgument("filter must have rank 5, and must compate to ref_image"));
    const auto filter_h = filter.dim_size(0);
    const auto filter_w = filter.dim_size(1);
    const auto filter_d = filter.dim_size(2);
    const auto out_channel_num = filter.dim_size(4);

    OP_REQUIRES(ctx, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == batch_size)
                      && (base_plane.dim_size(1) == image_height) && (base_plane.dim_size(2) == image_width) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to ref_image"));

    const auto out_height = (image_height + strides_[0] - 1)/strides_[0];
    const auto out_width = (image_width + strides_[1] - 1)/strides_[1];
    const auto out_depth = (image_depth + strides_[2] - 1)/strides_[2];
    OP_REQUIRES(ctx, (out_grad.shape().dims() == 5) && (out_grad.dim_size(0) == batch_size)
                      && (out_grad.dim_size(1) == out_height) && (out_grad.dim_size(2) == out_width) && (out_grad.dim_size(3) == out_depth)
                      && (out_grad.dim_size(4) == out_channel_num),
                errors::InvalidArgument("out_grad must have rank 5, and must compate to ref_image, got ", out_grad.shape().DebugString()));

    Tensor *images_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            images.shape(),
                            &images_grad));
    Tensor *filter_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1,
                            filter.shape(),
                            &filter_grad));
    Tensor *default_channels_value_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                              2,
                              default_channels_value.shape(),
                              &default_channels_value_grad));

  }
private:
  TF_DISALLOW_COPY_AND_ASSIGN(SparseConv3DFastGradOp);
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("SparseConv3DFastGrad") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          SparseConv3DFastGradOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
// TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

}  // namespace addons
}  // namespace tensorflow