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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "custom_helper_op/cc/kernels/sparse_conv3d_fast_op.h"

namespace tensorflow {
namespace custom_helper_op {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
__global__ void DeformableIm2ColKernel(
    int32 b, int32 num_kernels, SparseConv3DFastParams p,
    typename TTypes<T, 5>::Tensor input_eigen_tensor,
    typename TTypes<T, 8>::Tensor offset_eigen_tensor,
    typename TTypes<T, 7>::Tensor mask_eigen_tensor,
    typename TTypes<T, 4>::Tensor column_buffer_eigen_tensor) {
}

template <typename T>
__global__ void DeformableCol2ImForOffsetAndMaskKernel(
    int32 b, int32 num_kernels, SparseConv3DFastParams p,
    typename TTypes<T, 5>::Tensor input_eigen_tensor,
    typename TTypes<T, 8>::Tensor offset_eigen_tensor,
    typename TTypes<T, 7>::Tensor mask_eigen_tensor,
    typename TTypes<T, 4>::Tensor offset_grad_eigen_tensor,
    typename TTypes<T, 4>::Tensor mask_grad_eigen_tensor,
    typename TTypes<T, 6>::Tensor column_buffer_eigen_tensor) {
}

template <typename T>
__global__ void DeformableCol2ImForInputKernel(
    int32 b, int32 num_kernels, SparseConv3DFastParams p,
    typename TTypes<T, 8>::Tensor offset_eigen_tensor,
    typename TTypes<T, 7>::Tensor mask_eigen_tensor,
    typename TTypes<T, 4>::Tensor input_grad_eigen_tensor,
    typename TTypes<T, 1>::Tensor column_buffer_tensor_flattened) {
}

#define IM2COL(T)                                                              \
  template <>                                                                  \
  void SparseConv3DFastFunctorBase<GPUDevice, T>::DeformableIm2Col(            \
      OpKernelContext *context, int32 b) {                                     \
  }
TF_CALL_float(IM2COL);
TF_CALL_double(IM2COL);
#undef IM2COL

#define COL2IM_OFFSET_AND_MASK(T)                                              \
  template <>                                                                  \
  void                                                                         \
  SparseConv3DFastGradFunctor<GPUDevice, T>::DeformableCol2ImForOffsetAndMask( \
      OpKernelContext *context, int32 b) {                                     \
  }
TF_CALL_float(COL2IM_OFFSET_AND_MASK);
TF_CALL_double(COL2IM_OFFSET_AND_MASK);
#undef COL2IM_OFFSET_AND_MASK

#define COL2IM_INPUT(T)                                                        \
  template <>                                                                  \
  void SparseConv3DFastGradFunctor<GPUDevice, T>::DeformableCol2ImForInput(    \
      OpKernelContext *context, int32 b) {                                     \
  }
TF_CALL_float(COL2IM_INPUT);
TF_CALL_double(COL2IM_INPUT);
#undef COL2IM_INPUT

#define EXPLICIT_TEMPLATE(T)                                    \
  template struct SparseConv3DFastForwardFunctor<GPUDevice, T>; \
  template struct SparseConv3DFastGradFunctor<GPUDevice, T>;
TF_CALL_float(EXPLICIT_TEMPLATE);
TF_CALL_double(EXPLICIT_TEMPLATE);
#undef EXPLICIT_TEMPLATE

}  // end namespace functor

#define EXPLICIT_TEMPLATE(T)                   \
  template Status Transpose<GPUDevice, T, 5>(  \
      OpKernelContext * ctx, const Tensor &in, \
      const gtl::ArraySlice<int32> perm, Tensor *out);
TF_CALL_float(EXPLICIT_TEMPLATE);
TF_CALL_double(EXPLICIT_TEMPLATE);
#undef EXPLICIT_TEMPLATE

}  // namespace addons
}  // namespace tensorflow

#endif  // GOOGLE_CUDA