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

namespace functor {

#define SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST \
        int kKnownFilterHeight, int kKnownFilterWidth, int kKnownFilterDepth, \
        int kKnownDilationHeight, int kKnownDilationWidth, int kKnownDilationDepth, \
        int kKnownStrideHeight, int kKnownStrideWidth, int kKnownStrideDepth

#define SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST \
        kKnownFilterHeight, kKnownFilterWidth, kKnownFilterDepth, \
        kKnownDilationHeight, kKnownDilationWidth, kKnownDilationDepth, \
        kKnownStrideHeight, kKnownStrideWidth, kKnownStrideDepth

#define SPARSE_CONV3D_BASE_ARG_DEF_LIST \
              const int stride_h, \
              const int stride_w, \
              const int stride_d, \
              const int dilations_h, \
              const int dilations_w, \
              const int dilations_d, \
              const int filter_h, \
              const int filter_w, \
              const int filter_d, \
              const int batch_size, \
              const int image_height, \
              const int image_width, \
              const int image_depth, \
              const int image_channels, \
              const int out_height, \
              const int out_width, \
              const int out_depth, \
              const int out_channel_num, \
              const T* images_data, \
              const T* filter_data, \
              const T* default_channel_value, \
              const int32* base_plane_data



template <typename Device, typename T>
struct SparseConv3DFastFunctor {
void operator()(const Device& d, const SparseConv3DFastParams &p);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct SparseConv3DFastFunctor<Eigen::GpuDevice, T> {
void operator()(const Eigen::GpuDevice& d, const SparseConv3DFastParams &p);
};
#endif

template <typename Device, typename T>
struct SparseConv3DFastGradFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, 
              SPARSE_CONV3D_BASE_ARG_DEF_LIST,
              const T * out_grad_data,
              T * images_grad_data,
              T * filter_grad_data,
              T * default_channel_value_grad);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct SparseConv3DFastGradFunctor<Eigen::GpuDevice, T> {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Eigen::GpuDevice& d,
              SPARSE_CONV3D_BASE_ARG_DEF_LIST,
              const T * out_grad_data,
              T * images_grad_data,
              T * filter_grad_data,
              T * default_channel_value_grad);
};
#endif

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_