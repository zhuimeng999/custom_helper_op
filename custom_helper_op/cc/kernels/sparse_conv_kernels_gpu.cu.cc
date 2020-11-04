#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "custom_helper_op/cc/kernels/sparse_conv.h"

namespace tensorflow {
namespace custom_helper_op {
namespace functor {
// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;


// Zeroes count elements starting at ptr using all threads of a 1-D grid.
// Note: this function does not synchronize, and therefore the memory range is
// not guaranteed to be zero until the next kernel launch.
template <typename T, typename INDEX_TYPE>
__global__ void SparseConv2DKernel(const int32 count,
              const int stride_h,
              const int stride_w,
              const int dilation_h,
              const int dilations_w,
              const int filter_h,
              const int filter_w,
              const INDEX_TYPE batch_size, 
              const INDEX_TYPE image_height, 
              const INDEX_TYPE image_width,
              const INDEX_TYPE image_channels,
              const INDEX_TYPE out_channels,
              const T* images_data,
              const T* filter_data, 
              const T* base_plane_data,
              const T* default_channel_value,
              const T* offsets_data,
              T * out_data) {
  const auto image_height_step = image_width*image_channels;
  const auto image_batch_step = image_height*image_height_step;
  const auto filter_height_step = filter_w * image_channels;
  const auto filter_out_step = filter_h*filter_height_step;
  for (const auto i : GpuGridRangeX<INDEX_TYPE>(count)) {
    const auto batch_step = i/out_channels;
    const auto oc = i%out_channels;

    const auto tmp = batch_step/image_width;
    const auto w = batch_step%image_width;
    const auto b = tmp/image_height;
    const auto h = tmp%image_height;

    const auto image_start_h = h - filter_h/2;
    const auto image_start_w = w - filter_w/2;

    const auto image_batch_ptr = &images_data[b*image_batch_step];
    const auto base_plane_batch_ptr = &base_plane_data[b*image_height*image_width];
    const auto filter_out_channels_ptr = &filter_data[oc*filter_out_step];

    out_data[i] = T(0.);
    for(int f_h = 0; f_h < filter_h; f_h++){
      const auto im_h = image_start_h + f_h;
      const auto f_h_ptr = &filter_out_channels_ptr[f_h*filter_height_step];
      if((im_h >= 0) && (im_h < image_height)){
        const auto im_h_ptr = &image_batch_ptr[im_h*image_height_step];
        const auto base_plane_h_ptr = &base_plane_batch_ptr[im_h*image_width];
        for(int f_w = 0; f_w < filter_w; f_w++){
          const auto im_w = image_start_w + f_w;
          const auto f_w_ptr = &f_h_ptr[f_w*image_channels];
          if((im_w >= 0) && (im_w < image_width)){
            const auto im_w_ptr = &im_h_ptr[im_w*image_channels];
            T base_delta = base_plane_h_ptr[im_w] - base_plane_data[batch_step];
            for(int f_c = 0; f_c < image_channels; f_c++){
              const auto f_c_fixed = base_delta + f_c;
              if((f_c_fixed >= 0.) && (f_c_fixed < (image_channels - 1))){
                int f_c_new = static_cast<int>(f_c_fixed);
                T value_delta = f_c_fixed - f_c_new;
                T sample = im_w_ptr[f_c_new]*(1. - value_delta)+im_w_ptr[f_c_new + 1]*value_delta;
                out_data[i] += sample*f_w_ptr[f_c];
              } else {
                out_data[i] += default_channel_value[0]*f_w_ptr[f_c];
              }
            }
          } else {
            T tmp = T(0.);
            for(int f_c = 0; f_c < image_channels; f_c++){
              tmp += f_w_ptr[f_c];
            }
            out_data[i] += default_channel_value[0]*tmp;
          }
        }
      } else {
        T tmp = T(0.);
        for(int f_pos = 0; f_pos < filter_height_step; f_pos++){
          tmp += f_h_ptr[f_pos];
        }
        out_data[i] += default_channel_value[0]*tmp;
      }
    }
  }
}

#define SPARSE_CONV2D_ARG_LIST \
              loop_count, \
              stride_h, \ 
              stride_w, \
              dilation_h,\
              dilations_w,\
              filter_h,\
              filter_w,\
              batch_size,\ 
              image_height,\ 
              image_width,\
              image_channels,\
              out_channels,\
              images_data,\
              filter_data, \
              base_plane_data,\
              default_channel_value,\
              offsets_data,\
              out_data

template <typename T>
void SparseConv2DFunctor<Eigen::GpuDevice, T>::operator()(const Eigen::GpuDevice& d, 
              const int stride_h,
              const int stride_w,
              const int dilation_h,
              const int dilations_w,
              const int filter_h,
              const int filter_w,
              const int64 batch_size, 
              const int64 image_height, 
              const int64 image_width,
              const int64 image_channels,
              const int64 out_channels,
              const T* images_data,
              const T* filter_data, 
              const T* base_plane_data,
              const T* default_channel_value,
              const T* offsets_data,
              T * out_data)
{
  auto loop_count = batch_size*image_height*image_width*out_channels;
  if( loop_count > INT32_MAX){
    auto config = GetGpuLaunchConfig(loop_count, d, SparseConv2DKernel<T, int64>, 0, 0);
    SparseConv2DKernel<T, int64><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(SPARSE_CONV2D_ARG_LIST);  
  } else {
    auto config = GetGpuLaunchConfig(loop_count, d, SparseConv2DKernel<T, int32>, 0, 0);
    SparseConv2DKernel<T, int32><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(SPARSE_CONV2D_ARG_LIST);  
  }

};

template struct SparseConv2DFunctor<GPUDevice, float>;
template struct SparseConv2DFunctor<GPUDevice, double>;

template <typename T, typename INDEX_TYPE>
__global__ void SparseConv2DGradKernel(const int32 count,
              const int stride_h,
              const int stride_w,
              const int dilation_h,
              const int dilations_w,
              const int filter_h,
              const int filter_w,
              const INDEX_TYPE batch_size, 
              const INDEX_TYPE image_height, 
              const INDEX_TYPE image_width,
              const INDEX_TYPE image_channels,
              const INDEX_TYPE out_channels,
              const T* images_data,
              const T* filter_data, 
              const T* base_plane_data,
              const T* default_channel_value,
              const T* offsets_data,
              const T * out_grad_data,
              T* images_grad_data,
              T* filter_grad_data,
              T* base_plane_grad_data,
              T* default_channel_grad) {
  const auto image_height_step = image_width*image_channels;
  const auto image_batch_step = image_height*image_height_step;
  const auto filter_height_step = filter_w * image_channels;
  const auto filter_out_step = filter_h*filter_height_step;
  for (const auto i : GpuGridRangeX<INDEX_TYPE>(count)) {
    const auto batch_step = i/out_channels;
    const auto oc = i%out_channels;

    const auto tmp = batch_step/image_width;
    const auto w = batch_step%image_width;
    const auto b = tmp/image_height;
    const auto h = tmp%image_height;

    const auto image_start_h = h - filter_h/2;
    const auto image_start_w = w - filter_w/2;

    const auto image_batch_ptr = &images_data[b*image_batch_step];
    const auto images_grad_batch_ptr = &images_grad_data[b*image_batch_step];
    const auto filter_out_channels_ptr = &filter_data[oc*filter_out_step];
    const auto filter_out_channels_grad_ptr = &filter_grad_data[oc*filter_out_step];

    T out_grad = out_grad_data[i];
    for(int f_h = 0; f_h < filter_h; f_h++){
      const auto im_h = image_start_h + f_h;
      const auto f_h_ptr = &filter_out_channels_ptr[f_h*filter_height_step];
      const auto f_grad_h_ptr = &filter_out_channels_grad_ptr[f_h*filter_height_step];
      if((im_h >= 0) && (im_h < image_height)){
        const auto im_h_ptr = &image_batch_ptr[im_h*image_height_step];
        const auto im_grad_h_ptr = &images_grad_batch_ptr[im_h*image_height_step];
        for(int f_w = 0; f_w < filter_w; f_w++){
          const auto im_w = image_start_w + f_w;
          const auto f_w_ptr = &f_h_ptr[f_w*image_channels];
          const auto f_grad_w_ptr = &f_grad_h_ptr[f_w*image_channels];
          if((im_w >= 0) && (im_w < image_width)){
            const auto im_w_ptr = &im_h_ptr[im_w*image_channels];
            const auto im_grad_w_ptr = &im_grad_h_ptr[im_w*image_channels];
            for(int f_c = 0; f_c < image_channels; f_c++){
              atomicAdd(&im_grad_w_ptr[f_c], out_grad*f_w_ptr[f_c]);
              atomicAdd(&f_grad_w_ptr[f_c], out_grad*im_w_ptr[f_c]);
            }
          } else {
            T tmp = T(0.);
            T d_grad = out_grad*default_channel_value[0];
            for(int f_c = 0; f_c < image_channels; f_c++){
              tmp += f_w_ptr[f_c];
              atomicAdd(&f_grad_w_ptr[f_c], d_grad);
            }
            atomicAdd(&default_channel_grad[0], out_grad*tmp);
          }
        }
      } else {
        T tmp = T(0.);
        T d_grad = out_grad*default_channel_value[0];
        for(int f_pos = 0; f_pos < filter_height_step; f_pos++){
          tmp += f_h_ptr[f_pos];
          atomicAdd(&f_grad_h_ptr[f_pos], d_grad);
        }
        atomicAdd(&default_channel_grad[0], out_grad*tmp);
      }
    }
  }
}

#define SPARSE_CONV2D_GRAD_ARG_LIST \
              loop_count, \
              stride_h, \ 
              stride_w, \
              dilation_h,\
              dilations_w,\
              filter_h,\
              filter_w,\
              batch_size,\ 
              image_height,\ 
              image_width,\
              image_channels,\
              out_channels,\
              images_data,\
              filter_data, \
              base_plane_data,\
              default_channel_value,\
              offsets_data,\
              out_grad_data, \
              images_grad_data, \
              filter_grad_data, \ 
              base_plane_grad_data, \
              default_channel_value_grad 

// Zeroes count elements starting at ptr using all threads of a 1-D grid.
// Note: this function does not synchronize, and therefore the memory range is
// not guaranteed to be zero until the next kernel launch.
template <typename T, typename INDEX_TYPE>
__global__ void SetZeroBig(const INDEX_TYPE count, T* __restrict__ ptr) {
  for (const auto i : GpuGridRangeX<INDEX_TYPE>(count)) {
    ptr[i] = T(0);
  }
}

template <typename DeviceFunc>
GpuLaunchConfig GetGpuLaunchConfigBig(const int64 work_element_count,
                                   const Eigen::GpuDevice& d, DeviceFunc func,
                                   size_t dynamic_shared_memory_size,
                                   int block_size_limit) {
  CHECK_GT(work_element_count, 0);
  int block_count = 0;
  int thread_per_block = 0;

#if GOOGLE_CUDA
  cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
      &block_count, &thread_per_block, func, dynamic_shared_memory_size,
      block_size_limit);
  CHECK_EQ(err, cudaSuccess);
#elif TENSORFLOW_USE_ROCM
  // Earlier versions of this HIP routine incorrectly returned void.
  // TODO re-enable hipError_t error checking when HIP is fixed.
  // ROCm interface uses unsigned int, convert after checking
  uint32_t block_count_uint = 0;
  uint32_t thread_per_block_uint = 0;
  CHECK_GE(block_size_limit, 0);
  uint32_t block_size_limit_uint = static_cast<uint32_t>(block_size_limit);
  hipOccupancyMaxPotentialBlockSize(&block_count_uint, &thread_per_block_uint,
                                    func, dynamic_shared_memory_size,
                                    block_size_limit_uint);
  block_count = static_cast<int>(block_count_uint);
  thread_per_block = static_cast<int>(thread_per_block_uint);
#endif

  block_count =
      std::min(block_count, static_cast<int>((work_element_count + thread_per_block - 1) / thread_per_block));

  GpuLaunchConfig config;
  config.thread_per_block = thread_per_block;
  config.block_count = block_count;
  return config;
}

template <typename T>
void SparseConv2DGradFunctor<Eigen::GpuDevice, T>::operator()(const Eigen::GpuDevice& dev, 
              const int stride_h,
              const int stride_w,
              const int dilation_h,
              const int dilations_w,
              const int filter_h,
              const int filter_w,
              const int64 batch_size, 
              const int64 image_height, 
              const int64 image_width,
              const int64 image_channels,
              const int64 out_channels,
              const T* images_data,
              const T* filter_data, 
              const T* base_plane_data,
              const T* default_channel_value,
              const T* offsets_data,
              const T * out_grad_data,
              T * images_grad_data,
              T * filter_grad_data,
              T * base_plane_grad_data,
              T * default_channel_value_grad)
{
  auto base_plane_size = batch_size*image_height*image_width;
  auto images_size = batch_size*image_height*image_width*image_channels;
  auto filter_size = image_channels*out_channels*filter_h*filter_w;
  auto loop_count = batch_size*image_height*image_width*out_channels;
  if( loop_count > INT32_MAX){
    auto config = GetGpuLaunchConfigBig(images_size, dev, SetZeroBig<T, int64>, 0, 0);
    SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(images_size, images_grad_data);
    config = GetGpuLaunchConfigBig(filter_size, dev, SetZeroBig<T, int64>, 0, 0);
    SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(filter_size, filter_grad_data);
    config = GetGpuLaunchConfigBig(base_plane_size, dev, SetZeroBig<T, int64>, 0, 0);
    SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(base_plane_size, base_plane_grad_data);
    config = GetGpuLaunchConfigBig(1, dev, SetZeroBig<T, int64>, 0, 0);
    SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(1, default_channel_value_grad);

    config = GetGpuLaunchConfig(loop_count, dev, SparseConv2DGradKernel<T, int64>, 0, 0);
    SparseConv2DGradKernel<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(SPARSE_CONV2D_GRAD_ARG_LIST);  
  } else {
    auto config = GetGpuLaunchConfigBig(images_size, dev, SetZeroBig<T, int32>, 0, 0);
    SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(images_size, images_grad_data);
    config = GetGpuLaunchConfigBig(filter_size, dev, SetZeroBig<T, int32>, 0, 0);
    SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(filter_size, filter_grad_data);
    config = GetGpuLaunchConfigBig(base_plane_size, dev, SetZeroBig<T, int32>, 0, 0);
    SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(base_plane_size, base_plane_grad_data);
    config = GetGpuLaunchConfigBig(1, dev, SetZeroBig<T, int32>, 0, 0);
    SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(1, default_channel_value_grad);

    config = GetGpuLaunchConfig(loop_count, dev, SparseConv2DGradKernel<T, int32>, 0, 0);
    SparseConv2DGradKernel<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(SPARSE_CONV2D_GRAD_ARG_LIST);  
  }
}

template struct SparseConv2DGradFunctor<GPUDevice, float>;
template struct SparseConv2DGradFunctor<GPUDevice, double>;
} /* functor */
} /* custom_helper_op */
} /* tensorflow */

#endif /*EIGEN_USE_GPU */