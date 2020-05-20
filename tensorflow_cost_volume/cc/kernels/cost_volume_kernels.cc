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

template struct FillProjectiveTransform<CPUDevice, uint8>;
template struct FillProjectiveTransform<CPUDevice, int32>;
template struct FillProjectiveTransform<CPUDevice, int64>;
//template struct FillProjectiveTransform<CPUDevice, Eigen::half>;
template struct FillProjectiveTransform<CPUDevice, float>;
template struct FillProjectiveTransform<CPUDevice, double>;

template <typename Device, typename T>
void CostVolumeFunctor<Device, T>::operator() 
  (const Device& d, const Tensor& images, const Tensor& transforms, Tensor* output){
    const auto batch_size = output->dim_size(0);
    const auto image_height = output->dim_size(1);
    const auto image_width = output->dim_size(2);
    const auto image_depth = output->dim_size(3);
    const auto image_channels = output->dim_size(4);
  }

template struct CostVolumeFunctor<CPUDevice, uint8>;
template struct CostVolumeFunctor<CPUDevice, int32>;
template struct CostVolumeFunctor<CPUDevice, int64>;
template struct CostVolumeFunctor<CPUDevice, Eigen::half>;
template struct CostVolumeFunctor<CPUDevice, float>;
template struct CostVolumeFunctor<CPUDevice, double>;
}  // end namespace functor

typedef Eigen::ThreadPoolDevice CPUDevice;

using functor::CostVolumeFunctor;
using generator::Interpolation;
using generator::INTERPOLATION_BILINEAR;
using generator::INTERPOLATION_NEAREST;
using generator::ProjectiveGenerator;

template <typename Device, typename T>
class CostVolumeOp : public OpKernel {
 private:
  Interpolation interpolation_;

 public:
  explicit CostVolumeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    string interpolation_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("interpolation", &interpolation_str));
    if (interpolation_str == "NEAREST") {
      interpolation_ = INTERPOLATION_NEAREST;
    } else if (interpolation_str == "BILINEAR") {
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

    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            TensorShape({images_t.dim_size(0), images_t.dim_size(2), images_t.dim_size(3), transform_t.dim_size(2), images_t.dim_size(4)}),
                            &output_t));


    CostVolumeFunctor<Device, T>()(
        ctx->eigen_device<Device>(), images_t, transform_t, output_t);
  }
};

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("CostVolume") \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<TYPE>("dtype"),       \
                          CostVolumeOp<CPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
//TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// NOTE(ringwalt): We get an undefined symbol error if we don't explicitly
// instantiate the operator() in GCC'd code.
#define DECLARE_FUNCTOR(TYPE)                                               \
  template <>                                                               \
  void FillProjectiveTransform<GPUDevice, TYPE>::operator()(                \
      const GPUDevice& device, OutputType* output, const InputType& images, \
      const TransformsType& transform) const;                               \
  extern template struct FillProjectiveTransform<GPUDevice, TYPE>

TF_CALL_uint8(DECLARE_FUNCTOR);
TF_CALL_int32(DECLARE_FUNCTOR);
TF_CALL_int64(DECLARE_FUNCTOR);
TF_CALL_half(DECLARE_FUNCTOR);
TF_CALL_float(DECLARE_FUNCTOR);
TF_CALL_double(DECLARE_FUNCTOR);

}  // end namespace functor

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("CostVolume") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          CostVolumeOp<GPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
//TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA

}  // end namespace addons
}  // end namespace tensorflow
