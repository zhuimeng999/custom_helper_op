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

enum COST_REDUCE_METHOD { COST_REDUCE_MEAN, COST_REDUCE_MIN };

template <typename Device, typename T, bool half_centor>
struct CostVolumeFunctor {
  void operator()(const Device& dev, COST_REDUCE_METHOD reduce_method, 
              const int64 batch_size, 
              const int64 image_height, 
              const int64 image_width,
              const int64 image_channels,
              const int64 image_depth,
              const int64 src_image_num,
              const int64 src_image_height, 
              const int64 src_image_width,
              const int32 groups,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              T* cost_data,
              int32* cost_mask_data);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T, bool half_centor>
struct CostVolumeFunctor<Eigen::GpuDevice, T, half_centor> {
  void operator()(const Eigen::GpuDevice& dev, COST_REDUCE_METHOD reduce_method, 
               const int64 batch_size, 
              const int64 image_height, 
              const int64 image_width,
              const int64 image_channels,
              const int64 image_depth,
              const int64 src_image_num,
              const int64 src_image_height, 
              const int64 src_image_width,
              const int32 groups,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              T* cost_data,
              int32* cost_mask_data);
};
#endif

template <typename Device, typename T, bool half_centor>
struct CostVolumeGradFunctor {
  void operator()(    const Device& dev, COST_REDUCE_METHOD reduce_method, 
              const int64 batch_size, 
              const int64 image_height, 
              const int64 image_width,
              const int64 image_channels,
              const int64 image_depth,
              const int64 src_image_num,
              const int64 src_image_height, 
              const int64 src_image_width,
              const int32 groups,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              const T* cost_data,
              const int32* cost_mask_data,
              T* ref_image_grad_data,
              T* src_images_grad_data, 
              T* base_plane_grad_data
                                );
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T, bool half_centor>
struct CostVolumeGradFunctor<Eigen::GpuDevice, T, half_centor> {
  void operator()(    const Eigen::GpuDevice& dev, COST_REDUCE_METHOD reduce_method, 
              const int64 batch_size, 
              const int64 image_height, 
              const int64 image_width,
              const int64 image_channels,
              const int64 image_depth,
              const int64 src_image_num,
              const int64 src_image_height, 
              const int64 src_image_width,
              const int32 groups,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              const T* cost_data,
              const int32* cost_mask_data,
              T* ref_image_grad_data,
              T* src_images_grad_data, 
              T* base_plane_grad_data
                                );
};
#endif

}  // end namespace functor

}  // end namespace custom_helper_op
}  // end namespace tensorflow

#endif  // TENSORFLOW_ADDONS_IMAGE_KERNELS_IMAGE_PROJECTIVE_TRANSFORM_OP_H__
