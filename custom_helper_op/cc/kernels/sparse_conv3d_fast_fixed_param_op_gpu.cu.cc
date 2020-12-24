#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "custom_helper_op/cc/kernels/sparse_conv3d_fast_fixed_param_op.h"

namespace tensorflow {
namespace custom_helper_op {
namespace functor {
#if defined(CUB_PTX_WARP_THREADS)
  # define CUDA_WARP_SIZE CUB_PTX_WARP_THREADS
#else
  # define CUDA_WARP_SIZE 32
#endif

#define CHECK_PADDING(a, b) (static_cast<unsigned int>(a) >= static_cast<unsigned int>(b))

#define MY_TEMPLATE_INSTANCE(T, strideOnOutput, kKnownFilterRows, kKnownFilterCols, kKnownFilterDepths, \
                                                kKnownDilationRows, kKnownDilationCols, kKnownDilationDepths, \
                                                kKnownStrideRows, kKnownStrideCols, kKnownStrideDepths)

#define EXTERN_TEMPLATE_3(T, strideOnOutput, kKnownFilterRows, kKnownFilterCols, kKnownFilterDepths, \
                                    kKnownDilationRows, kKnownDilationCols, kKnownDilationDepths) \
          MY_TEMPLATE_INSTANCE(T, strideOnOutput, kKnownFilterRows, kKnownFilterCols, kKnownFilterDepths, \
                                    kKnownDilationRows, kKnownDilationCols, kKnownDilationDepths, 1, 1, 1) \
          MY_TEMPLATE_INSTANCE(T, strideOnOutput, kKnownFilterRows, kKnownFilterCols, kKnownFilterDepths, \
                                    kKnownDilationRows, kKnownDilationCols, kKnownDilationDepths, 2, 2, 2) \
          MY_TEMPLATE_INSTANCE(T, strideOnOutput, kKnownFilterRows, kKnownFilterCols, kKnownFilterDepths, \
                                    kKnownDilationRows, kKnownDilationCols, kKnownDilationDepths, 2, 2, 1)

#define EXTERN_TEMPLATE_2(T, strideOnOutput, kKnownFilterRows, kKnownFilterCols, kKnownFilterDepths) \
          EXTERN_TEMPLATE_3(T, strideOnOutput, kKnownFilterRows, kKnownFilterCols, kKnownFilterDepths, 1, 1, 1)

#define EXTERN_TEMPLATE_1(T, strideOnOutput) \
          EXTERN_TEMPLATE_2(T, strideOnOutput, 3, 3, 3)

#define EXTERN_TEMPLATE(T) \
          EXTERN_TEMPLATE_1(T, true)\
          EXTERN_TEMPLATE_1(T, false)

EXTERN_TEMPLATE(float)
EXTERN_TEMPLATE(double)
#undef MY_TEMPLATE_INSTANCE

// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;
// _Pragma("unroll") 
// GPU_DYNAMIC_SHARED_MEM_DECL
// Zeroes count elements starting at ptr using all threads of a 1-D grid.
// Note: this function does not synchronize, and therefore the memory range is
// not guaranteed to be zero until the next kernel launch.
template <typename T, bool strideOnOutput, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
__global__ void SparseConv3DFastFixedParamKernel(const SparseConv3DFastFixedParams p, const int64 count, const int kernel_step,
                                                                                          const T* images_data, 
                                                                                          const T* filter_data, 
                                                                                          const T* default_channel_value, 
                                                                                          const int32* base_plane_data,
                                                                                          T * output_data) 
{
  // GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(T), unsigned char, shared_memory);
  // T *shared_data = reinterpret_cast<T*>(shared_memory);

  const T default_value = ldg(default_channel_value);

  for(int64 i = ((blockIdx.x * blockDim.x + threadIdx.x)/CUDA_WARP_SIZE); i < count; i += (gridDim.x * (blockDim.x/CUDA_WARP_SIZE))){
    int32 out_ch  = i % p.output_channels;
    int32 depth   = i / p.output_channels % p.output_depths;
    int32 col     = i / p.output_channels / p.output_depths % p.output_cols;
    int32 row     = i / p.output_channels / p.output_depths / p.output_cols % p.output_rows;
    const int32 b = i / p.output_channels / p.output_depths / p.output_cols / p.output_rows;

    
    if(strideOnOutput){
      col = col * kKnownStrideCols;
      row = row * kKnownStrideRows;
      depth = (depth + ((ldg(base_plane_data + (b * p.input_rows + row) * p.input_cols + col) + kKnownStrideDepths - 1)/kKnownStrideDepths))*kKnownStrideDepths;
    } else {
      depth = depth + ldg(base_plane_data + (b * p.output_rows + row) * p.output_cols + col);
    }

    row = row - p.padding_rows;
    col = col - p.padding_cols;
    depth = depth - p.padding_depths;

    T partial_sum = T(0.);
    for(int j = threadIdx.x%CUDA_WARP_SIZE; j < kernel_step; j += CUDA_WARP_SIZE){
      int32 in_ch   = j % p.input_channels;
      int32 f_depth = j / p.input_channels % kKnownFilterDepths;
      int32 f_col   = j / p.input_channels / kKnownFilterDepths % kKnownFilterCols;
      int32 f_row   = j / p.input_channels / kKnownFilterDepths / kKnownFilterCols;

      int32 orign_row = row + f_row * kKnownDilationRows;
      int32 orign_col = col + f_col * kKnownDilationCols;

      T in_data = default_value;

      if(strideOnOutput){
        int64 image_index = (b * p.input_rows + orign_row) * p.input_cols + orign_col;
        bool is_padding = CHECK_PADDING(orign_row, p.input_rows) | CHECK_PADDING(orign_col, p.input_cols);

        if(!is_padding){
          int32 in_depth = depth + f_depth*kKnownDilationDepths - ldg(base_plane_data + image_index);
          is_padding = CHECK_PADDING(in_depth, p.input_depths);
          image_index = image_index * p.input_depths + in_depth;
          image_index = image_index * p.input_channels + in_ch;
          if(!is_padding){
            in_data = ldg(images_data + image_index);
          }
        }
      } else {
        int64 image_index = (b * p.output_rows + orign_row) * p.output_cols + orign_col;
        bool is_padding = CHECK_PADDING(orign_row, p.output_rows) | CHECK_PADDING(orign_col, p.output_cols);

        if(!is_padding){
          int32 orign_depth = depth + f_depth*kKnownDilationDepths;
          is_padding = ((orign_row % kKnownStrideRows) != 0) | ((orign_col % kKnownStrideCols) != 0) | ((orign_depth % kKnownStrideDepths) != 0);
          orign_depth = (orign_depth/kKnownStrideDepths) - (ldg(base_plane_data + image_index) + kKnownStrideDepths - 1)/kKnownStrideDepths ;
          is_padding |= CHECK_PADDING(orign_depth, p.input_depths);
          image_index = ((b * p.input_rows + orign_row/kKnownStrideRows) * p.input_cols + orign_col/kKnownStrideCols) * p.input_depths + orign_depth;
          image_index = image_index * p.input_channels + in_ch;
          if(!is_padding){
            in_data = ldg(images_data + image_index);
          }
        }
      }


      partial_sum += ldg(filter_data + out_ch*kernel_step + j) * in_data;
    }

  //   shared_data[threadIdx.x] = partial_sum;
  //   for (int offset = CUDA_WARP_SIZE/2; offset > 0; offset /= 2) {
  // #if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
  //       __syncwarp();
  // #endif
  //     if((threadIdx.x % CUDA_WARP_SIZE) < offset){
  //       shared_data[threadIdx.x] += shared_data[threadIdx.x + offset];
  //     }

  //   }
  //   __syncwarp();
  //   partial_sum = shared_data[threadIdx.x];

    for (int offset = CUDA_WARP_SIZE/2; offset > 0; offset /= 2) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
      partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, offset, 32);
#else
      partial_sum += __shfl_down(partial_sum, offset);
#endif
    }

    if((threadIdx.x%CUDA_WARP_SIZE) == 0){
      output_data[i] = partial_sum;
    }
  }
}

template <typename T, bool strideOnOutput, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
void SparseConv3DFastFixedParamFunctor<Eigen::GpuDevice, T, strideOnOutput, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>::operator()(const Eigen::GpuDevice& d, const SparseConv3DFastFixedParams p, 
                                               const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
                                                T * out_data)
{
  int64 loop_count = static_cast<int64>(p.input_batches)*p.output_rows*p.output_cols*p.output_depths*p.output_channels;
  int kernel_step = kKnownFilterRows * kKnownFilterCols * kKnownFilterDepths * p.input_channels;

  int block_per_grid = 0;
  int thread_per_block = 0;
  // CHECK_EQ(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&block_per_grid, &thread_per_block, SparseConv3DFastKernel<T, strideOnOutput>, [](int x){return x*sizeof(T);}), cudaSuccess);
  // CHECK_GT(block_per_grid, 0);
  // CHECK_EQ(thread_per_block%32, 0);

  // TF_CHECK_OK(GpuLaunchKernel(SparseConv3DFastKernel<T, strideOnOutput>, block_per_grid, 
  //                           thread_per_block, thread_per_block*sizeof(T), d.stream(), p, loop_count, kernel_step, images_data, filter_data, default_channel_value, base_plane_data, out_data));
  CHECK_EQ(cudaOccupancyMaxPotentialBlockSize(&block_per_grid, &thread_per_block, SparseConv3DFastFixedParamKernel<T, strideOnOutput, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>, 0), cudaSuccess);
  CHECK_GT(block_per_grid, 0);
  CHECK_EQ(thread_per_block%32, 0);

  TF_CHECK_OK(GpuLaunchKernel(SparseConv3DFastFixedParamKernel<T, strideOnOutput, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>, block_per_grid, 
                            thread_per_block, 0, d.stream(), p, loop_count, kernel_step, images_data, filter_data, default_channel_value, base_plane_data, out_data));

};

#define MY_TEMPLATE_INSTANCE(T, strideOnOutput, kKnownFilterHeight, kKnownFilterWidth, kKnownFilterDepth, \
                                                kKnownDilationHeight, kKnownDilationWidth, kKnownDilationDepth, \
                                                kKnownStrideHeight, kKnownStrideWidth, kKnownStrideDepth) \
          template struct SparseConv3DFastFixedParamFunctor<Eigen::GpuDevice, T, strideOnOutput, kKnownFilterHeight, kKnownFilterWidth, kKnownFilterDepth, \
                                                kKnownDilationHeight, kKnownDilationWidth, kKnownDilationDepth, \
                                                kKnownStrideHeight, kKnownStrideWidth, kKnownStrideDepth>;

EXTERN_TEMPLATE(float)
EXTERN_TEMPLATE(double)
#undef MY_TEMPLATE_INSTANCE

template <typename T, bool strideOnOutput, bool dynamic_default, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
__global__ void SparseConv3DFastFixedParamFilterGradKernel(const int32 count, const SparseConv3DFastFixedParams p, const int image_step,
                                                                                          const T* image_cnhwd,                                                                 
                                                                                          const T* filter_data, 
                                                                                          const T* default_channel_value, 
                                                                                          const int32* base_plane_data,
                                                                                          const T* out_grad_data, 
                                                                                          T * filter_grad_data,
                                                                                          T * default_channel_value_grad)
{
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(T), float, reduce_memory);

  T * shared_data = reinterpret_cast<T *>(reduce_memory);

  int out_ch = threadIdx.x%p.output_channels;
  int out_id_in_step = threadIdx.x/p.output_channels;
  int d_step = blockDim.x/p.output_channels;

  T default_value_grad = T(0.);
  const T default_value = ldg(default_channel_value);
  for(int i = blockIdx.x; i < count; i += gridDim.x){
    int depth = i % kKnownFilterDepths;
    int col   = i / kKnownFilterDepths % kKnownFilterCols;
    int row   = i / kKnownFilterDepths / kKnownFilterCols % kKnownFilterRows;
    int in_ch = i / kKnownFilterDepths / kKnownFilterCols / kKnownFilterRows;

    int filter_grad_out_channel_ptr = (((row*kKnownFilterCols + col)*kKnownFilterDepths + depth)*p.input_channels+in_ch)*p.output_channels;

    col = col * kKnownDilationCols - p.padding_cols;
    row = row * kKnownDilationRows - p.padding_rows;
    depth = depth * kKnownDilationDepths - p.padding_depths;
    
    T filter_grad = T(0.);
    for(int j = out_id_in_step; j < image_step; j += d_step){
      int im_d = j % p.output_depths;
      int im_c = j / p.output_depths % p.output_cols;
      int im_r = j / p.output_depths / p.output_cols % p.output_rows;
      int im_n = j / p.output_depths / p.output_cols / p.output_rows;
      
      bool is_padding;
      if(strideOnOutput){
        int out_d = ldg(base_plane_data + (im_n*p.input_rows + im_r * kKnownStrideRows)*p.input_cols+im_c * kKnownStrideCols);
        im_r = im_r * kKnownStrideRows + row;
        im_c = im_c * kKnownStrideCols + col;
        is_padding = (CHECK_PADDING(im_r, p.input_rows) | CHECK_PADDING(im_c, p.input_cols));
        if(!is_padding){
          im_d = (im_d + (out_d + kKnownStrideDepths - 1)/kKnownStrideDepths) * kKnownStrideDepths + depth 
                                              - ldg(base_plane_data + (im_n*p.input_rows + im_r)*p.input_cols+im_c);
          is_padding = CHECK_PADDING(im_d, p.input_depths);
        }
      } else {
        int out_d = ldg(base_plane_data + (im_n*p.output_rows + im_r)*p.output_cols+im_c);
        im_r = im_r + row;
        im_c = im_c + col;
        im_d = im_d + depth + out_d;
        is_padding = ((im_r % kKnownStrideRows) != 0) | ((im_c % kKnownStrideCols) != 0) | ((im_d % kKnownStrideDepths) != 0);
        int32 depth_index = (im_n*p.output_rows + im_r)*p.output_cols+im_c;

        im_r = im_r/kKnownStrideRows;
        im_c = im_c/kKnownStrideCols;
        is_padding |= (CHECK_PADDING(im_r, p.input_rows) | CHECK_PADDING(im_c, p.input_cols));
        if(!is_padding){
          im_d = (im_d/kKnownStrideDepths) - (ldg(base_plane_data + depth_index) + kKnownStrideDepths - 1)/kKnownStrideDepths ;
          is_padding = CHECK_PADDING(im_d, p.input_depths);
        }
      }

      T filter_grad_factor = default_value;
      if( is_padding ){
        if(dynamic_default){
          default_value_grad +=  filter_data[filter_grad_out_channel_ptr + out_ch]*ldg(out_grad_data + j*p.output_channels + out_ch);
        }
      } else {
        filter_grad_factor = ldg(image_cnhwd + (((in_ch * p.input_batches + im_n)*p.input_rows + im_r)*p.input_cols + im_c)*p.input_depths + im_d);
      }
      filter_grad += filter_grad_factor*ldg(out_grad_data + j*p.output_channels + out_ch);
    }

    shared_data[threadIdx.x] = filter_grad;
    __syncthreads();
    // if(threadIdx.x < p.output_channels){
    //   for(int k = out_ch + p.output_channels; k < blockDim.x; k += p.output_channels){
    //     filter_grad += shared_data[k];
    //   }
    //   filter_grad_data[filter_grad_out_channel_ptr + out_ch] = filter_grad;
    // }
    int is_odd = d_step%2;
    int offset = d_step/2;
    while(offset > 0){
      if(out_id_in_step < offset){
        shared_data[(is_odd + out_id_in_step) * p.output_channels + out_ch] += shared_data[(is_odd + out_id_in_step + offset) * p.output_channels + out_ch];
      }
      __syncthreads();
      offset = offset + is_odd;
      is_odd = offset%2;
      offset = offset/2;
    }
    if(threadIdx.x < p.output_channels){
      filter_grad_data[filter_grad_out_channel_ptr + threadIdx.x] = shared_data[threadIdx.x];
    }
    // __syncthreads(); /* ensure the shared data not modified by other quick threads */
  }

  if(dynamic_default){
    shared_data[threadIdx.x] = default_value_grad;
    __syncthreads();
    int is_odd = blockDim.x%2;
    int offset = blockDim.x/2;
    while(offset > 0){
      if(threadIdx.x < offset){
        shared_data[is_odd + threadIdx.x ] += shared_data[is_odd + threadIdx.x + offset];
      }
      __syncthreads();
      offset = offset + is_odd;
      is_odd = offset%2;
      offset = offset/2;
    }
    if(threadIdx.x == 0){
      GpuAtomicAdd(default_channel_value_grad, shared_data[0]);
    }
  }
}
template <typename T, bool strideOnOutput, bool dynamic_default, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
void SparseConv3DFastFixedParamFilterGradFunctor<Eigen::GpuDevice, T, strideOnOutput, dynamic_default, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>::operator()(const Eigen::GpuDevice& d, const SparseConv3DFastFixedParams p,
                                                const T* images_data, 
                                                const T* filter_data, 
                                                const T* default_channel_value, 
                                                const int32* base_plane_data,
                                                const T * out_grad_data,
                                                T * filter_grad_data,
                                                T * default_channel_value_grad)
{
  int32 loop_count = kKnownFilterRows*kKnownFilterCols*kKnownFilterDepths*p.input_channels;
  int image_step = p.input_batches*p.output_rows*p.output_cols*p.output_depths;

  int block_per_grid = 0;
  int thread_per_block = 0;
  CHECK_EQ(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&block_per_grid, &thread_per_block, SparseConv3DFastFixedParamFilterGradKernel<T, strideOnOutput, dynamic_default, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>, [](int x)->int{return x*sizeof(T);}), cudaSuccess);
  CHECK_GT(block_per_grid, 0);
  CHECK_GT(thread_per_block, p.output_channels);
  // CHECK_EQ(thread_per_block, 1024);
  thread_per_block = thread_per_block/p.output_channels*p.output_channels;

  TF_CHECK_OK(GpuLaunchKernel(SparseConv3DFastFixedParamFilterGradKernel<T, strideOnOutput, dynamic_default, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>, block_per_grid, thread_per_block, thread_per_block*sizeof(T), d.stream(), 
                                                                loop_count, p, image_step, 
                                                                images_data, filter_data, default_channel_value, base_plane_data, out_grad_data, 
                                                                filter_grad_data, default_channel_value_grad));
}

#define MY_TEMPLATE_INSTANCE(T, strideOnOutput, kKnownFilterHeight, kKnownFilterWidth, kKnownFilterDepth, \
                                                kKnownDilationHeight, kKnownDilationWidth, kKnownDilationDepth, \
                                                kKnownStrideHeight, kKnownStrideWidth, kKnownStrideDepth) \
          template struct SparseConv3DFastFixedParamFilterGradFunctor<Eigen::GpuDevice, T, strideOnOutput, true, kKnownFilterHeight, kKnownFilterWidth, kKnownFilterDepth, \
                                                kKnownDilationHeight, kKnownDilationWidth, kKnownDilationDepth, \
                                                kKnownStrideHeight, kKnownStrideWidth, kKnownStrideDepth>;\
          template struct SparseConv3DFastFixedParamFilterGradFunctor<Eigen::GpuDevice, T, strideOnOutput, false, kKnownFilterHeight, kKnownFilterWidth, kKnownFilterDepth, \
                                                kKnownDilationHeight, kKnownDilationWidth, kKnownDilationDepth, \
                                                kKnownStrideHeight, kKnownStrideWidth, kKnownStrideDepth>;

EXTERN_TEMPLATE(float)
EXTERN_TEMPLATE(double)
#undef MY_TEMPLATE_INSTANCE
} /* functor */
} /* custom_helper_op */
} /* tensorflow */

#endif /*EIGEN_USE_GPU */