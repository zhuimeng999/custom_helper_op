#ifndef _INDEX_INITIALIZER_H
#define _INDEX_INITIALIZER_H

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace custom_helper_op {
namespace functor {

#define SPARSE_PAD_BASE_ARG_DEF_LIST \
              const int stride_h, \
              const int stride_w, \
              const int stride_d, \
              const int batch_size, \
              const int image_height, \
              const int image_width, \
              const int image_depth, \
              const int image_channels, \
              const int out_height, \
              const int out_width, \
              const int out_depth, \
              const T* images_data, \
              const int32* base_plane_data

template <typename Device, typename T>
struct SparsePadFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(OpKernelContext* ctx, const Device& d, SPARSE_PAD_BASE_ARG_DEF_LIST, T* out_data);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct SparsePadFunctor<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* ctx, const Eigen::GpuDevice& d, SPARSE_PAD_BASE_ARG_DEF_LIST, T* out_data);
};
#endif

template <typename Device, typename T>
struct SparsePadGradFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(OpKernelContext* ctx, const Device& d, SPARSE_PAD_BASE_ARG_DEF_LIST, const T* out_grad_data, T* image_grad_data);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct SparsePadGradFunctor<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* ctx, const Eigen::GpuDevice& d, SPARSE_PAD_BASE_ARG_DEF_LIST, const T* out_grad_data, T* image_grad_data);
};
#endif

} /* functor */
} /* custom_helper_op */
} /* tensorflow */
#endif /* _INDEX_INITIALIZER_H */