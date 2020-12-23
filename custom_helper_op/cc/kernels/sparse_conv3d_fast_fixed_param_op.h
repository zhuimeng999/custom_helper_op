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

#ifndef CUSTOM_HELPER_OP_KERNELS_SPARSE_CONV3D_FAST_FIXED_PARAM_OP_H_
#define CUSTOM_HELPER_OP_KERNELS_SPARSE_CONV3D_FAST_FIXED_PARAM_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/batch_matmul_op_impl.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace custom_helper_op {

struct SparseConv3DFastFixedParams {
  int32 input_batches;
  int32 input_channels;
  int32 input_rows;
  int32 input_cols;
  int32 input_depths;
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
  bool dynamic_default;
};

#define SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST \
        int kKnownFilterHeight, int kKnownFilterWidth, int kKnownFilterDepth, \
        int kKnownDilationHeight, int kKnownDilationWidth, int kKnownDilationDepth, \
        int kKnownStrideHeight, int kKnownStrideWidth, int kKnownStrideDepth

#define SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST \
        kKnownFilterHeight, kKnownFilterWidth, kKnownFilterDepth, \
        kKnownDilationHeight, kKnownDilationWidth, kKnownDilationDepth, \
        kKnownStrideHeight, kKnownStrideWidth, kKnownStrideDepth

namespace functor {

template <typename Device, typename T, bool strideOnOutput, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
struct SparseConv3DFastFixedParamFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, const SparseConv3DFastFixedParams p,
                                                const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
                                                T * out_data);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T, bool strideOnOutput, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
struct SparseConv3DFastFixedParamFunctor<Eigen::GpuDevice, T, strideOnOutput, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST> {
void operator()(const Eigen::GpuDevice& d, const SparseConv3DFastFixedParams p,
                                                const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
                                                T * out_data);
};
#endif

template <typename Device, typename T, bool strideOnOutput, bool dynamic_default, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
struct SparseConv3DFastFixedParamFilterGradFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, const SparseConv3DFastFixedParams p,
                                                const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
                                                const T * out_grad_data,
                                                T * filter_grad_data,
                                                T * default_channel_value_grad);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T, bool strideOnOutput, bool dynamic_default, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
struct SparseConv3DFastFixedParamFilterGradFunctor<Eigen::GpuDevice, T, strideOnOutput, dynamic_default, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST> {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Eigen::GpuDevice& d, const SparseConv3DFastFixedParams p,
                                                const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
                                                const T * out_grad_data,
                                                T * filter_grad_data,
                                                T * default_channel_value_grad);
};
#endif

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_