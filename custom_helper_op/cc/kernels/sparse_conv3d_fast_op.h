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

#ifndef CUSTOM_HELPER_OP_KERNELS_SPARSE_CONV3D_FAST_OP_H_
#define CUSTOM_HELPER_OP_KERNELS_SPARSE_CONV3D_FAST_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/batch_matmul_op_impl.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace custom_helper_op {

struct SparseConv3DFastParams {
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

template <typename Device, typename T>
Status TensorSetZero(OpKernelContext *ctx, Tensor *value);

template <typename Device, typename T, int NDIMS>
struct LaunchTransposeAndReverse {

bool operator()(OpKernelContext *ctx, const Tensor &in,
                 const gtl::ArraySlice<int32> perm, const gtl::ArraySlice<bool> reverse, Tensor *out);
};

namespace functor {

template <typename Device, typename T, bool strideOnOutput>
struct SparseConv3DFastFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, const SparseConv3DFastParams p,
                                                const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
                                                T * out_data);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T, bool strideOnOutput>
struct SparseConv3DFastFunctor<Eigen::GpuDevice, T, strideOnOutput> {
void operator()(const Eigen::GpuDevice& d, const SparseConv3DFastParams p,
                                                const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
                                                T * out_data);
};
#endif

template <typename Device, typename T>
struct SparseConv3DFastGradFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, const SparseConv3DFastParams p,
                                                const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
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
void operator()(const Eigen::GpuDevice& d, const SparseConv3DFastParams p,
                                                const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
                                                const T * out_grad_data,
                                                T * images_grad_data,
                                                T * filter_grad_data,
                                                T * default_channel_value_grad);
};
#endif

template <typename Device, typename T, bool strideOnOutput, bool dynamic_default>
struct SparseConv3DFastFilterGradFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, const SparseConv3DFastParams p,
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
template <typename T, bool strideOnOutput, bool dynamic_default>
struct SparseConv3DFastFilterGradFunctor<Eigen::GpuDevice, T, strideOnOutput, dynamic_default> {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Eigen::GpuDevice& d, const SparseConv3DFastParams p,
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