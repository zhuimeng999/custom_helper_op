#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "custom_helper_op/cc/kernels/index_initializer.h"

namespace tensorflow {
namespace custom_helper_op {
namespace functor {
// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;


// Zeroes count elements starting at ptr using all threads of a 1-D grid.
// Note: this function does not synchronize, and therefore the memory range is
// not guaranteed to be zero until the next kernel launch.
template <typename T>
__global__ void SetToIndex(const int32 count, T* __restrict__ out_data, const int32 out_width) {
  for (const auto i : GpuGridRangeX<int32>(count)) {
    T *tmp = &out_data[i*3];
    tmp[0] = T(i%out_width);
    tmp[1] = T(i/out_width);
    tmp[2] = T(1.0);
  }
}

template <typename T>
void FillIndexFunctor<Eigen::GpuDevice, T>::operator()(OpKernelContext* ctx, const Eigen::GpuDevice& d, T *out_data,int32 out_height, int32 out_width)
{
  auto total_elm = out_height*out_width;
  auto config = GetGpuLaunchConfig(total_elm, d, SetToIndex<T>, 0, 0);
  SetToIndex<T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(total_elm, out_data, out_width);  
};

template struct FillIndexFunctor<GPUDevice, float>;

} /* functor */
} /* custom_helper_op */
} /* tensorflow */

#endif /*EIGEN_USE_GPU */