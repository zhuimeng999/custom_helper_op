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
    const Tensor& images_t = ctx->input(0);
    const Tensor& transform_t = ctx->input(1);
    OP_REQUIRES(ctx, images_t.shape().dims() == 5,
                errors::InvalidArgument("Input images must have rank 4"));
    OP_REQUIRES(ctx, (transform_t.shape().dims() == 4) && (transform_t.dim_size(3) == 8),
                errors::InvalidArgument("Input transform must have rank 4, and the last dim must be 8"));
    OP_REQUIRES(ctx, (transform_t.dim_size(0) == images_t.dim_size(0)) && (transform_t.dim_size(1) == (images_t.dim_size(1) - 1)),
                errors::InvalidArgument("the first dim of images and transforms must be equal, and the second dim of images and transforms must greater than 1"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            TensorShape({images_t.dim_size(0), images_t.dim_size(2), images_t.dim_size(3), transform_t.dim_size(2), images_t.dim_size(4)}),
                            &output));
    Tensor* output_mask = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1,
                            TensorShape({images_t.dim_size(0), images_t.dim_size(2), images_t.dim_size(3), transform_t.dim_size(2), 1}),
                            &output_mask));

    CostAggregateFunctor<Device, T>()(
          ctx->eigen_device<Device>(), images_t, transform_t, output, output_mask);

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
