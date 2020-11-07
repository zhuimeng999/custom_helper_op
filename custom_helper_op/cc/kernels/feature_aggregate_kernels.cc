/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "custom_helper_op/cc/kernels/feature_aggregate.h"

namespace tensorflow {
namespace custom_helper_op {

#define COST_FUNCTOR_ARG_LIST \
                                  batch_size, \
                                  image_height, \
                                  image_width,\
                                  image_channels,\
                                  image_depth,\
                                  src_image_num,\
                                  src_image_height, \
                                  src_image_width,\
                                  src_images.tensor<T, 5>().data(), \
                                  base_plane.tensor<T, 4>().data(),\
                                  offsets.tensor<T, 2>().data(),\
                                  Rs.tensor<T, 4>().data(),\
                                  Ts.tensor<T, 3>().data(),\
                                  mapped_feature->tensor<T, 6>().data(),\
                                  mapped_mask->tensor<int32, 6>().data()


using functor::FeatureAggregateFunctor;
using functor::FeatureAggregateGradFunctor;

template <typename Device, typename T>
class FeatureAggregateOp : public OpKernel {
  private:
  bool half_centor_;
 public:
  explicit FeatureAggregateOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("half_centor", &half_centor_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& src_images = ctx->input(0);
    const Tensor& base_plane = ctx->input(1);
    const Tensor& offsets = ctx->input(2);
    const Tensor& Rs = ctx->input(3);
    const Tensor& Ts = ctx->input(4);

    OP_REQUIRES(ctx, src_images.shape().dims() == 5,
                errors::InvalidArgument("src image must have rank 5"));

    const auto batch_size = src_images.dim_size(0);
    const auto src_image_num = src_images.dim_size(1);
    const auto src_image_height = src_images.dim_size(2);
    const auto src_image_width = src_images.dim_size(3);
    const auto image_channels = src_images.dim_size(4);

    OP_REQUIRES(ctx, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == batch_size) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to src_images"));
    const auto image_height = base_plane.dim_size(1);
    const auto image_width = base_plane.dim_size(2);

    OP_REQUIRES(ctx, (offsets.shape().dims() == 2) && (offsets.dim_size(0) == batch_size),
                errors::InvalidArgument("offsets must have rank 2, and must compate to src_images"));
    const auto image_depth = offsets.dim_size(1);

    OP_REQUIRES(ctx, (Rs.shape().dims() == 4) && (Rs.dim_size(0) == batch_size)
                      && (Rs.dim_size(1) == src_image_num) && (Rs.dim_size(2) == 3) && (Rs.dim_size(3) == 3),
                errors::InvalidArgument("Rs must have rank 2, and must compate to src_images"));
    OP_REQUIRES(ctx, (Ts.shape().dims() == 3) && (Ts.dim_size(0) == batch_size)
                      && (Ts.dim_size(1) == src_image_num) && (Ts.dim_size(2) == 3),
                errors::InvalidArgument("Ts must have rank 2, and must compate to src_images"));

    Tensor* mapped_feature = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            TensorShape({batch_size, image_height, image_width, image_depth, src_image_num, image_channels}),
                            &mapped_feature));
    Tensor* mapped_mask = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1,
                            TensorShape({batch_size, image_height, image_width, image_depth, src_image_num, 1}),
                            &mapped_mask));

    if(half_centor_){
      FeatureAggregateFunctor<Device, T, true>()(
                                    ctx->eigen_device<Device>(),
                                    COST_FUNCTOR_ARG_LIST);
    } else {
      FeatureAggregateFunctor<Device, T, false>()(
                                    ctx->eigen_device<Device>(),
                                    COST_FUNCTOR_ARG_LIST);
    }

  }
private:
  TF_DISALLOW_COPY_AND_ASSIGN(FeatureAggregateOp);
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("FeatureAggregate") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          FeatureAggregateOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA

#define COST_FUNCTOR_GRAD_ARG_LIST \
                                  batch_size, \
                                  image_height, \
                                  image_width,\
                                  image_channels,\
                                  image_depth,\
                                  src_image_num,\
                                  src_images.dim_size(2), \
                                  src_images.dim_size(3),\
                                  src_images.tensor<T, 5>().data(), \
                                  base_plane.tensor<T, 4>().data(),\
                                  offsets.tensor<T, 2>().data(),\
                                  Rs.tensor<T, 4>().data(),\
                                  Ts.tensor<T, 3>().data(),\
                                  mapped_feature_grad.tensor<T, 6>().data(),\
                                  mapped_mask.tensor<int32, 6>().data(),\
                                  src_images_grad->tensor<T, 5>().data(),\
                                  base_plane_grad->tensor<T, 4>().data()

template <typename Device, typename T>
class FeatureAggregateGradOp : public OpKernel {
private:
 bool half_centor_;

 public:
  explicit FeatureAggregateGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("half_centor", &half_centor_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& src_images = ctx->input(0);
    const Tensor& base_plane = ctx->input(1);
    const Tensor& offsets = ctx->input(2);
    const Tensor& Rs = ctx->input(3);
    const Tensor& Ts = ctx->input(4);
    const Tensor& mapped_feature_grad = ctx->input(5);
    const Tensor& mapped_mask = ctx->input(6);

    OP_REQUIRES(ctx, src_images.shape().dims() == 5,
                errors::InvalidArgument("src image must have rank 5"));

    const auto batch_size = src_images.dim_size(0);
    const auto src_image_num = src_images.dim_size(1);
    const auto src_image_height = src_images.dim_size(2);
    const auto src_image_width = src_images.dim_size(3);
    const auto image_channels = src_images.dim_size(4);

    OP_REQUIRES(ctx, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == batch_size) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to src_images"));
    const auto image_height = base_plane.dim_size(1);
    const auto image_width = base_plane.dim_size(2);

    OP_REQUIRES(ctx, (offsets.shape().dims() == 2) && (offsets.dim_size(0) == batch_size),
                errors::InvalidArgument("offsets must have rank 2, and must compate to src_images"));
    const auto image_depth = offsets.dim_size(1);

    OP_REQUIRES(ctx, (Rs.shape().dims() == 4) && (Rs.dim_size(0) == batch_size)
                      && (Rs.dim_size(1) == src_image_num) && (Rs.dim_size(2) == 3) && (Rs.dim_size(3) == 3),
                errors::InvalidArgument("Rs must have rank 2, and must compate to src_images"));
    OP_REQUIRES(ctx, (Ts.shape().dims() == 3) && (Ts.dim_size(0) == batch_size)
                      && (Ts.dim_size(1) == src_image_num) && (Ts.dim_size(2) == 3),
                errors::InvalidArgument("Ts must have rank 3, and must compate to src_images"));

    OP_REQUIRES(ctx, (mapped_feature_grad.shape().dims() == 6) && (mapped_feature_grad.dim_size(0) == batch_size)
                      && (mapped_feature_grad.dim_size(1) == image_height) && (mapped_feature_grad.dim_size(2) == image_width)
                      && (mapped_feature_grad.dim_size(3) == image_depth) && (mapped_feature_grad.dim_size(4) == src_image_num) && (mapped_feature_grad.dim_size(5) == image_channels),
                errors::InvalidArgument("mapped_feature_grad must have rank 6, and must compate to src_images, got ", mapped_feature_grad.shape().DebugString()));

    OP_REQUIRES(ctx, (mapped_mask.shape().dims() == 6) && (mapped_mask.dim_size(0) == batch_size)
                      && (mapped_mask.dim_size(1) == image_height) && (mapped_mask.dim_size(2) == image_width)
                      && (mapped_mask.dim_size(3) == image_depth) && (mapped_mask.dim_size(4) == src_image_num) && (mapped_mask.dim_size(5) == 1),
                errors::InvalidArgument("mapped_mask must have rank 6, and must compate to src_images"));

    Tensor* src_images_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            src_images.shape(),
                            &src_images_grad));

     Tensor* base_plane_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1,
                            base_plane.shape(),
                            &base_plane_grad));                           

    if(half_centor_){
      FeatureAggregateGradFunctor<Device, T, true>()(
                                    ctx->eigen_device<Device>(),
                                    COST_FUNCTOR_GRAD_ARG_LIST);
    } else {
      FeatureAggregateGradFunctor<Device, T, false>()(
                                    ctx->eigen_device<Device>(),
                                    COST_FUNCTOR_GRAD_ARG_LIST);
    }

  }
private:
  TF_DISALLOW_COPY_AND_ASSIGN(FeatureAggregateGradOp);
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("FeatureAggregateGrad") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          FeatureAggregateGradOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA
}  // end namespace addons
}  // end namespace tensorflow
