#ifndef _INDEX_INITIALIZER_H
#define _INDEX_INITIALIZER_H

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace custom_helper_op {
namespace functor {

template <typename Device, typename T, bool half_centor>
struct SparseConv2DFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(OpKernelContext* ctx, const Device& d, T *out_data,int32 out_height, int32 out_width);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T, bool half_centor>
struct SparseConv2DFunctor<Eigen::GpuDevice, T, half_centor> {
  void operator()(OpKernelContext* ctx, const Eigen::GpuDevice& d, T *out_data,int32 out_height, int32 out_width);
};
#endif

} /* functor */
} /* custom_helper_op */
} /* tensorflow */
#endif /* _INDEX_INITIALIZER_H */