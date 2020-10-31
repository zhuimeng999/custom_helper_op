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

#ifndef TENSORFLOW_ADDONS_IMAGE_KERNELS_IMAGE_PROJECTIVE_TRANSFORM_OP_H__
#define TENSORFLOW_ADDONS_IMAGE_KERNELS_IMAGE_PROJECTIVE_TRANSFORM_OP_H__

// See docs in ../ops/image_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace custom_helper_op {

// NOTE(ringwalt): We MUST wrap the generate() call in a functor and explicitly
// instantiate the functor in image_ops_gpu.cu.cc. Otherwise, we will be missing
// some Eigen device code.
namespace functor {

template <typename Device, typename T>
struct CostAggregateFunctor {
  void operator()(const Device& d, const Tensor& images, const Tensor& transforms, Tensor* output, Tensor *output_mask);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct CostAggregateFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, const Tensor& images, const Tensor& transforms, Tensor* output, Tensor *output_mask);
};
#endif

template <typename Device, typename T>
struct CostAggregateGradFunctor {
  void operator()(const Device& d, const Tensor& images, const Tensor& transforms, const Tensor& transformed_mask, const Tensor& grad, Tensor* output);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct CostAggregateGradFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, const Tensor& images, const Tensor& transforms, const Tensor& transformed_mask, const Tensor& grad, Tensor* output);
};
#endif

}  // end namespace functor

}  // end namespace custom_helper_op
}  // end namespace tensorflow

#endif  // TENSORFLOW_ADDONS_IMAGE_KERNELS_IMAGE_PROJECTIVE_TRANSFORM_OP_H__
