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
#include "tensorflow_cost_volume/cc/kernels/cost_volume.h"

namespace tensorflow {
namespace addons {

namespace functor {

// Explicit instantiation of the CPU functor.
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, Interpolation INTERPOLATION_TYPE>
void CostVolumeFunctor<Device, T, INTERPOLATION_TYPE>::operator() 
  (const Device& d, const Tensor& images, const Tensor& transforms, Tensor* output){
    CHECK_EQ(1, 2);
  }

  template struct CostVolumeFunctor<CPUDevice, float, INTERPOLATION_BILINEAR>;

  template <typename Device, typename T, Interpolation INTERPOLATION_TYPE>
  void CostVolumeGradFunctor<Device, T, INTERPOLATION_TYPE>::operator() 
    (const Device& d, const Tensor& images, const Tensor& transforms, const Tensor& grad, Tensor* output){
      CHECK_EQ(1, 2);
  }
  template struct CostVolumeGradFunctor<CPUDevice, float, INTERPOLATION_BILINEAR>;
}  // end namespace functor

typedef Eigen::ThreadPoolDevice CPUDevice;

using functor::CostVolumeFunctor;
using functor::CostVolumeGradFunctor;
using functor::Interpolation;
using functor::INTERPOLATION_BILINEAR;
using functor::INTERPOLATION_NEAREST;

template <typename Device, typename T>
class CostVolumeOp : public OpKernel {
 private:
  Interpolation interpolation_;

 public:
  explicit CostVolumeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    string interpolation_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("interpolation", &interpolation_str));
    if (interpolation_str == "BILINEAR") {
      interpolation_ = INTERPOLATION_BILINEAR;
    } else {
      LOG(FATAL) << "Invalid interpolation " << interpolation_str
                 << ". Supported types: NEAREST, BILINEAR";
    }
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

    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            TensorShape({images_t.dim_size(0), images_t.dim_size(2), images_t.dim_size(3), transform_t.dim_size(2), images_t.dim_size(4)}),
                            &output_t));


    if(interpolation_ == INTERPOLATION_BILINEAR){
      CostVolumeFunctor<Device, T, INTERPOLATION_BILINEAR>()(
          ctx->eigen_device<Device>(), images_t, transform_t, output_t);
    } 

  }
};

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("CostVolume") \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<TYPE>("dtype"),       \
                          CostVolumeOp<CPUDevice, TYPE>)

TF_CALL_float(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("CostVolume") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          CostVolumeOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA


template <typename Device, typename T>
class CostVolumeGradOp : public OpKernel {
 private:
  Interpolation interpolation_;

 public:
  explicit CostVolumeGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    string interpolation_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("interpolation", &interpolation_str));
    if (interpolation_str == "BILINEAR") {
      interpolation_ = INTERPOLATION_BILINEAR;
    } else {
      LOG(FATAL) << "Invalid interpolation " << interpolation_str
                 << ". Supported types: NEAREST, BILINEAR";
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& images_t = ctx->input(0);
    const Tensor& transform_t = ctx->input(1);
    const Tensor& grad_t = ctx->input(2);
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

    if(interpolation_ == INTERPOLATION_BILINEAR){
      CostVolumeGradFunctor<Device, T, INTERPOLATION_BILINEAR>()(
          ctx->eigen_device<Device>(), images_t, transform_t, grad_t, output_t);
    } 
  }
};

  #define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("CostVolumeGrad") \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<TYPE>("dtype"),       \
                          CostVolumeGradOp<CPUDevice, TYPE>)

  TF_CALL_float(REGISTER);

  #undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("CostVolumeGrad") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          CostVolumeGradOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA
}  // end namespace addons
}  // end namespace tensorflow
