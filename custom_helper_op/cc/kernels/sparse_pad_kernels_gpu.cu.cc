#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "custom_helper_op/cc/kernels/sparse_pad.h"

namespace tensorflow {
namespace custom_helper_op {
namespace functor {
// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;

#define SPARSE_PAD_BASE_ARG_LIST \
              stride_h, \
              stride_w, \
              stride_d, \
              batch_size, \
              image_height, \
              image_width, \
              image_depth, \
              image_channels, \
              out_height, \
              out_width, \
              out_depth, \
              images_data, \
              base_plane_data

// Zeroes count elements starting at ptr using all threads of a 1-D grid.
// Note: this function does not synchronize, and therefore the memory range is
// not guaranteed to be zero until the next kernel launch.
template <typename T, typename INDEX_TYPE>
__global__ void SparsePadKernel(const INDEX_TYPE count, SPARSE_PAD_BASE_ARG_DEF_LIST, T* out_data) {
  for (const auto i : GpuGridRangeX<INDEX_TYPE>(count)) {
    int32 d = i%out_depth;
    INDEX_TYPE depth_map_pos = i/out_depth;
 
    int32 w = depth_map_pos%out_width;
    const int32 tmp = depth_map_pos/out_width;

    int32 h = tmp%out_height;
    const int32 b = tmp/out_height;

    auto out_channel_ptr = &out_data[i*image_channels];

    d = d + base_plane_data[depth_map_pos] - ((base_plane_data[depth_map_pos] + stride_d - 1)/stride_d)*stride_d;
    if(((h%stride_h) == 0) && ((w%stride_w) == 0) && ((d%stride_d) == 0)){
      d = d/stride_d;
      const auto image_channel_ptr = &images_data[(((b*image_height + (h/stride_h))*image_width + (w/stride_w))*image_depth + d)*image_channels];
      for(auto c = 0; c < image_channels; c++){
        out_channel_ptr[c] = image_channel_ptr[c];
      }
    } else {
      for(auto c = 0; c < image_channels; c++){
        out_channel_ptr[c] = T(0.);
      }
    }
  }
}

template <typename T>
void SparsePadFunctor<Eigen::GpuDevice, T>::operator()(OpKernelContext* ctx, const Eigen::GpuDevice& d, SPARSE_PAD_BASE_ARG_DEF_LIST, T* out_data)
{
  auto total_elm = batch_size*out_height*out_width*out_depth;
  auto config = GetGpuLaunchConfig(total_elm, d, SparsePadKernel<T, int32>, 0, 0);
  SparsePadKernel<T, int32><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(total_elm, SPARSE_PAD_BASE_ARG_LIST, out_data);  
};

template struct SparsePadFunctor<GPUDevice, float>;
template struct SparsePadFunctor<GPUDevice, double>;

// Zeroes count elements starting at ptr using all threads of a 1-D grid.
// Note: this function does not synchronize, and therefore the memory range is
// not guaranteed to be zero until the next kernel launch.
template <typename T, typename INDEX_TYPE>
__global__ void SparsePadGradKernel(const INDEX_TYPE count, SPARSE_PAD_BASE_ARG_DEF_LIST, const T* out_grad_data, T* image_grad_data) {
  for (const auto i : GpuGridRangeX<INDEX_TYPE>(count)) {
    int32 d = i%image_depth;
    INDEX_TYPE depth_map_pos = i/image_depth;
 
    int32 w = depth_map_pos%image_width;
    const int32 tmp = depth_map_pos/image_width;

    int32 h = tmp%image_height;
    const int32 b = tmp/image_height;

    auto image_grad_channel_ptr = &image_grad_data[i*image_channels];

    depth_map_pos = (b*out_height+h)*out_width+w;

    h = h * stride_h;
    w = w * stride_w;
    d = d * stride_d + ((base_plane_data[depth_map_pos] + stride_d - 1)/stride_d)*stride_d - base_plane_data[depth_map_pos];
    auto out_grad_channel_ptr = &out_grad_data[(depth_map_pos*out_depth + d)*image_channels];
    for(auto c = 0; c < image_channels; c++){
      image_grad_channel_ptr[c] = out_grad_channel_ptr[c];
    }
  }
}

template <typename T>
void SparsePadGradFunctor<Eigen::GpuDevice, T>::operator()(OpKernelContext* ctx, const Eigen::GpuDevice& d, SPARSE_PAD_BASE_ARG_DEF_LIST, const T* out_grad_data, T* image_grad_data)
{
  const auto loop_count = static_cast<int64>(batch_size)*image_height*image_width*image_depth;
  auto config = GetGpuLaunchConfig(loop_count, d, SparsePadGradKernel<T, int32>, 0, 0);
  SparsePadGradKernel<T, int32><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(loop_count, SPARSE_PAD_BASE_ARG_LIST, out_grad_data, image_grad_data);  
};

template struct SparsePadGradFunctor<GPUDevice, float>;
template struct SparsePadGradFunctor<GPUDevice, double>;
} /* functor */
} /* custom_helper_op */
} /* tensorflow */

#endif /*EIGEN_USE_GPU */