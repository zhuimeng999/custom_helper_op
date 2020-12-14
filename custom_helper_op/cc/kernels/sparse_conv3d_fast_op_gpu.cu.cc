#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "custom_helper_op/cc/kernels/sparse_conv3d_fast_op.h"
#include "gpu/cub/device/device_reduce.cuh"

namespace tensorflow {
namespace custom_helper_op {

template struct LaunchTransposeAndReverse<Eigen::GpuDevice, float, 5>;
template struct LaunchTransposeAndReverse<Eigen::GpuDevice, double, 5>;

namespace functor {
#define CHECK_PADDING(a, b) (static_cast<unsigned int>(a) >= static_cast<unsigned int>(b))

// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;
// _Pragma("unroll") 
// GPU_DYNAMIC_SHARED_MEM_DECL
// Zeroes count elements starting at ptr using all threads of a 1-D grid.
// Note: this function does not synchronize, and therefore the memory range is
// not guaranteed to be zero until the next kernel launch.
template <typename T, int BLOCK_THREADS>
__global__ void SparseConv3DFastKernel(const SparseConv3DFastParams p, const int64 count, const int kernel_step,
                                                                                          const T* images_data, 
                                                                                          const T* filter_data, 
                                                                                          const T* default_channel_value, 
                                                                                          const int32* base_plane_data,
                                                                                          T * output_data) 
{
  // Specialize WarpReduce for type int
  typedef cub::WarpReduce<T> WarpReduce;
  // Allocate WarpReduce shared memory for 4 warps
  __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_THREADS/CUB_PTX_WARP_THREADS];
  // int64 loop_index = blockIdx.x * blockDim.x + threadIdx.x;

  for(int64 i = ((blockIdx.x * BLOCK_THREADS + threadIdx.x)/CUB_PTX_WARP_THREADS); i < count; i += (gridDim.x * (BLOCK_THREADS/CUB_PTX_WARP_THREADS))){
    int32 out_ch = i % p.output_channels;
    int64 tmp = i / p.output_channels;

    int32 depth   = tmp   % p.output_depths;
    tmp = tmp   / p.output_depths;

    int32 col   = tmp % p.output_cols;
    tmp       = tmp / p.output_cols;

    int32 row   = tmp % p.output_rows;
    const int32 b   = tmp / p.output_rows;

    col = col * p.stride_cols;
    row = row * p.stride_rows;

    depth = (depth + ((__ldg(base_plane_data + (b * p.input_rows + row) * p.input_cols + col) + p.stride_depths - 1)/p.stride_depths))*p.stride_depths;

    row = row - p.padding_rows;
    col = col - p.padding_cols;
    depth = depth - p.padding_depths;

    T partial_sum = T(0.);
    for(int j = threadIdx.x%CUB_PTX_WARP_THREADS; j < kernel_step; j += CUB_PTX_WARP_THREADS){
      int32 in_ch = j % p.input_channels;
      int32 f_tmp = j / p.input_channels;
      int32 f_depth = f_tmp % p.filter_depths;
      f_tmp = f_tmp/p.filter_depths;
      int32 f_col = f_tmp % p.filter_cols;
      int32 f_row = f_tmp/p.filter_cols;

      int32 in_tmp = row + f_row*p.dilation_rows ;
      bool is_padding = CHECK_PADDING(in_tmp, p.input_rows);
      int64 image_index = b * p.input_rows + in_tmp;
      in_tmp = col + f_col*p.dilation_cols;
      is_padding |= CHECK_PADDING(in_tmp, p.input_cols);
      image_index = image_index * p.input_cols + in_tmp;
      if(!is_padding){
        in_tmp = depth + f_depth*p.dilation_depths - __ldg(base_plane_data + image_index);
        is_padding = CHECK_PADDING(in_tmp, p.input_depths);
        image_index = image_index * p.input_depths + in_tmp;
      }
      image_index = image_index * p.input_channels + in_ch;

      T in_data = is_padding?__ldg(default_channel_value):__ldg(images_data + image_index);

      partial_sum += __ldg(filter_data + out_ch*kernel_step + j) * in_data;
    }

    T total_sum = WarpReduce(temp_storage[threadIdx.x/CUB_PTX_WARP_THREADS]).Sum(partial_sum);
    if((threadIdx.x%CUB_PTX_WARP_THREADS) == 0){
      output_data[i] = total_sum;
    }
  }
}

template <typename T>
void SparseConv3DFastFunctor<Eigen::GpuDevice, T>::operator()(const Eigen::GpuDevice& d, const SparseConv3DFastParams p, 
                                               const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
                                                T * out_data)
{
  int block_count = 0;
  int thread_per_block = 1024;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &block_count, SparseConv3DFastKernel<T, 1024>, 1024, 0);
  if((err != cudaSuccess) || (block_count <= 0)){
    thread_per_block = 512;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &block_count, SparseConv3DFastKernel<T, 512>, 512, 0);
      
      if((err != cudaSuccess) || (block_count <= 0)){
        thread_per_block = 256;
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &block_count, SparseConv3DFastKernel<T, 256>, 256, 0);
      }
  }
  CHECK_EQ(err, cudaSuccess);
  CHECK_GT(block_count, 0);
  
  int64 loop_count = static_cast<int64>(p.input_batches)*p.output_rows*p.output_cols*p.output_depths*p.output_channels;
  int valid_sample_per_block = thread_per_block/CUB_PTX_WARP_THREADS;
  int64 expect_total_block_count = (loop_count + valid_sample_per_block - 1)/valid_sample_per_block;
  block_count = std::min(static_cast<int64>(d.getNumGpuMultiProcessors() * block_count), expect_total_block_count);

  int kernel_step = p.filter_rows * p.filter_cols * p.filter_depths * p.input_channels;
  switch (thread_per_block)
  {
  case 1024:
    TF_CHECK_OK(GpuLaunchKernel(SparseConv3DFastKernel<T, 1024>, block_count, 
                              1024, 0, d.stream(), p, loop_count, kernel_step, images_data, filter_data, default_channel_value, base_plane_data, out_data));
    break;
  case 512:
    TF_CHECK_OK(GpuLaunchKernel(SparseConv3DFastKernel<T, 512>, block_count, 
                              512, 0, d.stream(), p, loop_count, kernel_step, images_data, filter_data, default_channel_value, base_plane_data, out_data));
  break;
  case 256:
    TF_CHECK_OK(GpuLaunchKernel(SparseConv3DFastKernel<T, 256>, block_count, 
                              256, 0, d.stream(), p, loop_count, kernel_step, images_data, filter_data, default_channel_value, base_plane_data, out_data));
    break;
  default:
    break;
  }
};

template struct SparseConv3DFastFunctor<Eigen::GpuDevice, float>;
template struct SparseConv3DFastFunctor<Eigen::GpuDevice, double>;

template <typename T, int BLOCK_THREADS>
__global__ void SparseConv3DFastGradKernel(const SparseConv3DFastParams p, const int64 count, const int kernel_step,
                                                                                          const T* out_grad_data, 
                                                                                          const T* filter_data, 
                                                                                          const T* default_channel_value, 
                                                                                          const int32* base_plane_data,
                                                                                          T * image_grad_data) 
{
  // Specialize WarpReduce for type int
  typedef cub::WarpReduce<T> WarpReduce;
  // Allocate WarpReduce shared memory for 4 warps
  __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_THREADS/CUB_PTX_WARP_THREADS];
  // int64 loop_index = blockIdx.x * blockDim.x + threadIdx.x;

  for(int64 i = ((blockIdx.x * BLOCK_THREADS + threadIdx.x)/CUB_PTX_WARP_THREADS); i < count; i += (gridDim.x * (BLOCK_THREADS/CUB_PTX_WARP_THREADS))){
    int32 in_ch = i % p.input_channels;
    int64 tmp = i / p.input_channels;

    int32 depth   = tmp % p.input_depths;
    tmp = tmp   / p.input_depths;

    int32 col   = tmp % p.input_cols;
    tmp       = tmp / p.input_cols;

    int32 row   = tmp % p.input_rows;
    const int32 b   = tmp / p.input_rows;

    depth = depth + __ldg(base_plane_data + (b * p.input_rows + row) * p.input_cols + col) + p.padding_depths;
    row = row + p.padding_rows;
    col = col + p.padding_cols;

    T partial_sum = T(0.);
    for(int j = threadIdx.x%CUB_PTX_WARP_THREADS; j < kernel_step; j += CUB_PTX_WARP_THREADS){
      int32 out_ch = j % p.output_channels;
      int32 f_tmp = j / p.output_channels;
      int32 f_depth = f_tmp % p.filter_depths;
      f_tmp = f_tmp/p.filter_depths;
      int32 f_col = f_tmp % p.filter_cols;
      int32 f_row = f_tmp/p.filter_cols;

      int32 in_tmp = row - f_row*p.dilation_rows;
      int32 base_plane_index = b * p.input_rows + in_tmp;
      bool is_padding = (in_tmp%p.stride_rows != 0);
      in_tmp = in_tmp/p.stride_rows;
      is_padding = is_padding || CHECK_PADDING(in_tmp, p.output_rows);
      int64 image_index = b * p.output_rows + in_tmp;

      in_tmp = col - f_col*p.dilation_cols;
      base_plane_index = base_plane_index * p.input_cols + in_tmp;
      is_padding |= (in_tmp%p.stride_cols != 0);
      in_tmp = in_tmp/p.stride_cols;
      is_padding |= CHECK_PADDING(in_tmp, p.output_cols);
      image_index = image_index * p.output_cols + in_tmp;
      if(!is_padding){
        /* check here to prevent out of memory error when read base_plane_data */
        in_tmp = depth - f_depth*p.dilation_depths;
        is_padding = (in_tmp%p.stride_depths != 0);
        in_tmp = (in_tmp/p.stride_depths) - (__ldg(base_plane_data + base_plane_index) + p.stride_depths - 1)/p.stride_depths ;
        is_padding |= CHECK_PADDING(in_tmp, p.output_depths);
        image_index = image_index * p.output_depths + in_tmp;
      }
      image_index = image_index * p.output_channels + out_ch;

      T in_data = is_padding?T(0):__ldg(out_grad_data + image_index);

      partial_sum += __ldg(filter_data + in_ch*kernel_step + j) * in_data;
    }

    T total_sum = WarpReduce(temp_storage[threadIdx.x/CUB_PTX_WARP_THREADS]).Sum(partial_sum);
    if((threadIdx.x%CUB_PTX_WARP_THREADS) == 0){
      image_grad_data[i] = total_sum;
    }
  }
}


template <typename T>
__global__ void SparseConv3DFastFilterGradKernel(const SparseConv3DFastParams p, const int64 count, const int image_size,
                                                                                          const T* image_cnhwd,
                                                                                          const T* out_grad_data,                                                                                         
                                                                                          const T* filter_data, 
                                                                                          const T* default_channel_value, 
                                                                                          const int32* base_plane_data,
                                                                                          T * filter_grad_data)
{
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(T), unsigned char, shared_memory);

  T * shared_data = reinterpret_cast<T *>(shared_memory);

  assert((blockDim.x % p.output_channels) == 0);
  int out_ch = threadIdx.x%p.output_channels;
  int d_offset = threadIdx.x/p.output_channels;
  for(int i = blockIdx.x; i < count; i += gridDim.x){
    int depth = i % p.filter_depths;
    int col = i / p.filter_depths % p.filter_cols;
    int row = i / p.filter_depths/ p.filter_cols % p.filter_rows;
    int in_ch = i / p.filter_depths/ p.filter_cols /p.filter_rows;

    int filter_grad_out_channel_ptr = (((row*p.filter_cols + col)*p.filter_depths + depth)*p.input_channels+in_ch)*p.output_channels;
    depth = depth - p.padding_depths;
    col = col - p.padding_cols;
    row = row - p.padding_rows;

    int image_index_offset = (row * p.input_cols+ col)*p.input_depths + depth;
    T filter_grad = T(0.);
    for(int j = threadIdx.x; j < image_size; j += d_offset){
      int im_d = j % p.input_depths + depth;
      int im_c = j / p.input_depths % p.input_cols + col;
      int im_r = j / p.input_depths / p.input_cols % p.input_rows + row;
      
      if(CHECK_PADDING(im_r, p.input_rows) || CHECK_PADDING(im_c, p.input_cols) || CHECK_PADDING(im_d, p.input_depths)){

      } else {
        filter_grad += image_cnhwd[j - image_index_offset]*out_grad_data[j*p.output_channels + out_ch];
      }

    }
    shared_data[threadIdx.x] = filter_grad;
    __syncthreads();
    if(threadIdx.x < p.output_channels){
      for(int k = out_ch + p.output_channels; k < blockDim.x; k += p.output_channels){
        filter_grad += shared_data[k];
      }
      filter_grad_data[filter_grad_out_channel_ptr+ out_ch] = filter_grad;
    }
  }
}
#include <iostream>
// Partially specialize functor for GpuDevice.
template <typename T>
void SparseConv3DFastGradFunctor<Eigen::GpuDevice, T>::operator()(const Eigen::GpuDevice& d, const SparseConv3DFastParams p,
                                                const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
                                                const T * out_grad_data,
                                                T * images_grad_data,
                                                T * filter_grad_data,
                                                T * default_channel_value_grad)
{
  int block_count = 0;
  int thread_per_block = 1024;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &block_count, SparseConv3DFastGradKernel<T, 1024>, 1024, 0);
  if((err != cudaSuccess) || (block_count <= 0)){
    thread_per_block = 512;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &block_count, SparseConv3DFastGradKernel<T, 512>, 512, 0);
      
      if((err != cudaSuccess) || (block_count <= 0)){
        thread_per_block = 256;
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &block_count, SparseConv3DFastGradKernel<T, 256>, 256, 0);
      }
  }
  CHECK_EQ(err, cudaSuccess);
  CHECK_GT(block_count, 0);
  
  int64 loop_count = static_cast<int64>(p.input_batches)*p.input_rows*p.input_cols*p.input_depths*p.input_channels;
  int valid_sample_per_block = thread_per_block/CUB_PTX_WARP_THREADS;
  int64 expect_total_block_count = (loop_count + valid_sample_per_block - 1)/valid_sample_per_block;
  block_count = std::min(static_cast<int64>(d.getNumGpuMultiProcessors() * block_count), expect_total_block_count);

  int kernel_step = p.filter_rows * p.filter_cols * p.filter_depths * p.output_channels;
  switch (thread_per_block)
  {
  case 1024:
    TF_CHECK_OK(GpuLaunchKernel(SparseConv3DFastGradKernel<T, 1024>, block_count, 
                              1024, 0, d.stream(), p, loop_count, kernel_step, out_grad_data, filter_data, default_channel_value, base_plane_data, images_grad_data));
    break;
  case 512:
    TF_CHECK_OK(GpuLaunchKernel(SparseConv3DFastGradKernel<T, 512>, block_count, 
                              512, 0, d.stream(), p, loop_count, kernel_step, out_grad_data, filter_data, default_channel_value, base_plane_data, images_grad_data));
  break;
  case 256:
    TF_CHECK_OK(GpuLaunchKernel(SparseConv3DFastGradKernel<T, 256>, block_count, 
                              256, 0, d.stream(), p, loop_count, kernel_step, out_grad_data, filter_data, default_channel_value, base_plane_data, images_grad_data));
    break;
  default:
    break;
  }

  // size_t smem_size = 10*10*4;
  // err = cudaOccupancyMaxPotentialBlockSize(&block_count, &thread_per_block, SparseConv3DFastFilterGradKernel<T>, smem_size);
  // CHECK_EQ(err, cudaSuccess);
  // CHECK_GT(block_count, 0);
  // std::cout << "1####### " << block_count << " " << thread_per_block << " " << smem_size << std::endl;
  // smem_size = 0;
  // err = cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, SparseConv3DFastFilterGradKernel<T>, block_count/d.getNumGpuMultiProcessors(), thread_per_block);
  // CHECK_EQ(err, cudaSuccess);
  // std::cout << "2####### " << block_count << " " << thread_per_block << " " << smem_size << " " << d.getNumGpuMultiProcessors()<< std::endl;
};

template struct SparseConv3DFastGradFunctor<Eigen::GpuDevice, float>;
template struct SparseConv3DFastGradFunctor<Eigen::GpuDevice, double>;

} /* functor */
} /* custom_helper_op */
} /* tensorflow */

#endif /*EIGEN_USE_GPU */