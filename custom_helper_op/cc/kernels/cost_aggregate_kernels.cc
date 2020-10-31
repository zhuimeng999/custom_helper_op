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
#include "custom_helper_op/cc/kernels/cost_aggregate.h"

namespace tensorflow {
namespace custom_helper_op {

using functor::CostAggregateFunctor;
using functor::CostAggregateGradFunctor;

template <typename Device, typename T>
class CostAggregateOp : public OpKernel {
 public:
  explicit CostAggregateOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& ref_image = ctx->input(0);
    const Tensor& src_images = ctx->input(1);
    const Tensor& base_plane = ctx->input(2);
    const Tensor& offsets = ctx->input(3);
    const Tensor& Rs = ctx->input(4);
    const Tensor& Ts = ctx->input(5);

    const auto batch_size = ref_image.dim_size(0);
    const auto image_height = ref_image.dim_size(1);
    const auto image_width = ref_image.dim_size(2);
    const auto image_channels = ref_image.dim_size(3);
    const auto src_image_num = src_images.dim_size(1);
    const auto image_depth = offsets.dim_size(1);
    OP_REQUIRES(ctx, ref_image.shape().dims() == 4,
                errors::InvalidArgument("ref image must have rank 4"));
    OP_REQUIRES(ctx, (src_images.shape().dims() == 5) && (src_images.dim_size(0) == batch_size)
                      && (src_images.dim_size(2) == image_height) && (src_images.dim_size(3) == image_width) && (src_images.dim_size(4) == image_channels),
                errors::InvalidArgument("src_images must have rank 5, and must compate to ref_image"));
    OP_REQUIRES(ctx, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == batch_size)
                      && (base_plane.dim_size(1) == image_height) && (base_plane.dim_size(2) == image_width) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to ref_image"));
    OP_REQUIRES(ctx, (offsets.shape().dims() == 2) && (offsets.dim_size(0) == batch_size),
                errors::InvalidArgument("offsets must have rank 2, and must compate to ref_image"));
    OP_REQUIRES(ctx, (Rs.shape().dims() == 4) && (Rs.dim_size(0) == batch_size)
                      && (Rs.dim_size(1) == src_image_num) && (Rs.dim_size(2) == 3) && (Rs.dim_size(3) == 3),
                errors::InvalidArgument("Rs must have rank 2, and must compate to ref_image"));
    OP_REQUIRES(ctx, (Ts.shape().dims() == 3) && (Ts.dim_size(0) == batch_size)
                      && (Ts.dim_size(1) == src_image_num) && (Ts.dim_size(2) == 3),
                errors::InvalidArgument("Ts must have rank 2, and must compate to ref_image"));
     
    const auto output_shape = TensorShape({batch_size, image_height, image_width, image_depth});
    Tensor* cost = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            output_shape,
                            &cost));
    Tensor* cost_mask = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1,
                            output_shape,
                            &cost_mask));

    CostAggregateFunctor<Device, T>()(
                                ctx->eigen_device<Device>(), 
                                batch_size, 
                                image_height, 
                                image_width,
                                image_channels,
                                image_depth,
                                src_image_num,
                                src_images.dim_size(2), 
                                src_images.dim_size(3),
                                ref_image.tensor<T, 4>().data(),
                                src_images.tensor<T, 5>().data(), 
                                base_plane.tensor<T, 4>().data(),
                                offsets.tensor<T, 2>().data(),
                                Rs.tensor<T, 4>().data(),
                                Ts.tensor<T, 3>().data(),
                                cost->tensor<T, 4>().data(),
                                cost_mask->tensor<int32, 4>().data());

  }
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("CostAggregate") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          CostAggregateOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA


template <typename Device, typename T>
class CostAggregateGradOp : public OpKernel {
 public:
  explicit CostAggregateGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& images_t = ctx->input(0);
    const Tensor& transform_t = ctx->input(1);
    const Tensor& transformed_mask = ctx->input(2);
    const Tensor& grad_t = ctx->input(3);
    OP_REQUIRES(ctx, images_t.shape().dims() == 5,
                errors::InvalidArgument("Input images must have rank 5"));
    OP_REQUIRES(ctx, (transform_t.shape().dims() == 4) && (transform_t.dim_size(3) == 8),
                errors::InvalidArgument("Input transform must have rank 4, and the last dim must be 8"));
    OP_REQUIRES(ctx, (transform_t.dim_size(0) == images_t.dim_size(0)) && (transform_t.dim_size(1) == (images_t.dim_size(1) - 1)),
                errors::InvalidArgument("the first dim of images and transforms must be equal, and the second dim of images and transforms must greater than 1"));

    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            images_t.shape(),
                            &output_t));

    CostAggregateGradFunctor<Device, T>()(
          ctx->eigen_device<Device>(), images_t, transform_t, transformed_mask, grad_t, output_t);
    
  }
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("CostAggregateGrad") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          CostAggregateGradOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA
}  // end namespace addons
}  // end namespace tensorflow
