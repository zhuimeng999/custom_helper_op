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

#define SPARSE_CONV3D_KERNEL_BASE_ARG_DEF_LIST \
              const int stride_h, \
              const int stride_w, \
              const int stride_d, \
              const int dilations_h_arg, \
              const int dilations_w_arg, \
              const int dilations_d_arg, \
              const int filter_h_arg, \
              const int filter_w_arg, \
              const int filter_d_arg, \
              const int batch_size, \
              const int image_height, \
              const int image_width, \
              const int image_depth, \
              const int image_channels, \
              const int out_channel_num, \
              const T* images_data, \
              const T* filter_data, \
              const T* default_channel_value, \
              const int32* base_plane_data
 

#define SPARSE_CONV3D_KERNEL_BASE_ARG_LIST \
              stride_h, \
              stride_w, \
              stride_d, \
              dilations_h, \
              dilations_w, \
              dilations_d, \
              filter_h, \
              filter_w, \
              filter_d, \
              batch_size, \
              image_height, \
              image_width, \
              image_depth, \
              image_channels, \
              out_channel_num, \
              images_data, \
              filter_data, \
              default_channel_value, \
              base_plane_data


// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;
// _Pragma("unroll") 
// GPU_DYNAMIC_SHARED_MEM_DECL
// Zeroes count elements starting at ptr using all threads of a 1-D grid.
// Note: this function does not synchronize, and therefore the memory range is
// not guaranteed to be zero until the next kernel launch.
template <typename T, typename INDEX_TYPE, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
__global__ void SparseConv3DKernel(const INDEX_TYPE count, SPARSE_CONV3D_KERNEL_BASE_ARG_DEF_LIST, T * out_data) {
  // GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(T), unsigned char , shared_memory);
  // T* const shared_data = reinterpret_cast<T*>(shared_memory);
  const int filter_h = kKnownFilterHeight < 0 ? filter_h_arg: kKnownFilterHeight;
  const int filter_w = kKnownFilterWidth < 0 ? filter_w_arg: kKnownFilterWidth;
  const int filter_d = kKnownFilterDepth < 0 ? filter_d_arg: kKnownFilterDepth;
  const int dilations_h = kKnownDilationHeight < 0 ? dilations_h_arg: kKnownDilationHeight;
  const int dilations_w = kKnownDilationWidth < 0 ? dilations_w_arg: kKnownDilationWidth;
  const int dilations_d = kKnownDilationDepth < 0 ? dilations_d_arg: kKnownDilationDepth;

  const INDEX_TYPE image_width_step = image_depth*image_channels;
  const INDEX_TYPE image_height_step = image_width*image_width_step;
  const INDEX_TYPE image_batch_step = image_height*image_height_step;
  const auto filter_depth_step = image_channels*out_channel_num;
  const auto filter_width_step = filter_d*filter_depth_step;
  const auto filter_height_step = filter_w*filter_width_step;
  for (const auto i : GpuGridRangeX<INDEX_TYPE>(count)) {
    const int32 d = i%image_depth;
    const INDEX_TYPE depth_map_pos = i/image_depth;
 
    const int32 w = depth_map_pos%image_width;
    const int32 tmp = depth_map_pos/image_width;

    const int32 h = tmp%image_height;
    const int32 b = tmp/image_height;

    const auto image_start_h = h - (filter_h/2)*dilations_h;
    const auto image_start_w = w - (filter_w/2)*dilations_w;
    const auto image_start_d = d - (filter_d/2)*dilations_d;

    const auto image_batch_ptr = &images_data[b*image_batch_step];
    const auto base_plane_batch_ptr = &base_plane_data[b*image_height*image_width];

    auto out_channels = &out_data[i*out_channel_num];

    for(int o_c = 0; o_c < out_channel_num; o_c++){
      out_channels[o_c] = 0.;
    }
    
    _Pragma("unroll")  for(int f_h = 0; f_h < filter_h; f_h++){
      const auto im_h = image_start_h + f_h*dilations_h;
      const auto f_h_ptr = &filter_data[f_h*filter_height_step];
      if((im_h >= 0) && (im_h < image_height)){
        /* 1. valid height pixel */
        const auto im_h_ptr = &image_batch_ptr[im_h*image_height_step];
        const auto base_plane_h_ptr = &base_plane_batch_ptr[im_h*image_width];
        _Pragma("unroll")   for(int f_w = 0; f_w < filter_w; f_w++){
          const auto im_w = image_start_w + f_w*dilations_w;
          const auto f_w_ptr = &f_h_ptr[f_w*filter_width_step];
          if((im_w >= 0) && (im_w < image_width)){
            /* 2. valid width pixel */
            const auto im_w_ptr = &im_h_ptr[im_w*image_width_step];
            const auto base_delta_d = image_start_d + base_plane_data[depth_map_pos] - base_plane_h_ptr[im_w];
            _Pragma("unroll")   for(int f_d = 0; f_d < filter_d; f_d++){
              const auto im_d = base_delta_d + f_d * dilations_d;
              const auto f_d_ptr = &f_w_ptr[f_d*filter_depth_step];
              if((im_d >= 0) && (im_d < image_depth)){
                /* 3. valid depth pixel */
                const auto im_d_ptr = &im_w_ptr[im_d*image_channels];
                for(int f_c = 0; f_c < image_channels; f_c++){
                  const auto f_o_ptr = &f_d_ptr[f_c*out_channel_num];
                  for(int o_c = 0; o_c < out_channel_num; o_c++){
                    /* output channel loop */
                    out_channels[o_c] += f_o_ptr[o_c]*im_d_ptr[f_c];
                  }
                }
              } else {
                /* 3. empty depth pixel */
                for(int32 f_pos = 0; f_pos < filter_depth_step; f_pos = f_pos + out_channel_num){
                  const auto f_o_ptr = &f_d_ptr[f_pos];
                  for(int o_c = 0; o_c < out_channel_num; o_c++){
                    out_channels[o_c] += default_channel_value[0]*f_o_ptr[o_c];
                  }
                }
              }
            }
          } else {
            /* 2. empty width pixel */
            for(int32 f_pos = 0; f_pos < filter_width_step; f_pos = f_pos + out_channel_num){
              const auto f_o_ptr = &f_w_ptr[f_pos];
              for(int o_c = 0; o_c < out_channel_num; o_c++){
                out_channels[o_c] += default_channel_value[0]*f_o_ptr[o_c];
              }
            }
          }
        }
      } else {
        /* 1. empty height pixel */
        for(int32 f_pos = 0; f_pos < filter_height_step; f_pos = f_pos + out_channel_num){
          const auto f_o_ptr = &f_h_ptr[f_pos];
          for(int o_c = 0; o_c < out_channel_num; o_c++){
            out_channels[o_c] += default_channel_value[0]*f_o_ptr[o_c];
          }
        }
      }
    }
  }
}

#define SPARSE_CONV3D_KERNEL_ARG_LIST loop_count, SPARSE_CONV3D_KERNEL_BASE_ARG_LIST, out_data

template <typename T, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
void SparseConv3DFunctor<Eigen::GpuDevice, T, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>::operator()(const Eigen::GpuDevice& d, 
              SPARSE_CONV3D_BASE_ARG_DEF_LIST,
              T * out_data)
{
  const auto loop_count = static_cast<int64>(batch_size)*image_height*image_width*image_depth;
  const auto images_size = static_cast<int64>(batch_size)*image_height*image_width*image_depth*image_channels;
  const auto out_size = static_cast<int64>(batch_size)*image_height*image_width*image_depth*out_channel_num;
  if( (images_size > INT32_MAX) | (out_size > INT32_MAX)){
    auto config = GetGpuLaunchConfig(loop_count, d, SparseConv3DKernel<T, int64, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>, 0, 0);
    SparseConv3DKernel<T, int64, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(SPARSE_CONV3D_KERNEL_ARG_LIST);  
  } else {
    auto config = GetGpuLaunchConfig(loop_count, d, SparseConv3DKernel<T, int32, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>, 0, 0);
    SparseConv3DKernel<T, int32, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(SPARSE_CONV3D_KERNEL_ARG_LIST);  
  }

};


#define SPARSE_CONV3D_DEFINE_INSTANCE(instance_type, f_h, f_w, f_d, d_h, d_w, d_d) \
                template struct SparseConv3DFunctor<GPUDevice, instance_type, f_h, f_w, f_d, d_h, d_w, d_d>;


#define SPARSE_CONV3D_DEFINE_INSTANCE_WITH_DILATIONS(instance_type, d_h, d_w, d_d) \
                SPARSE_CONV3D_DEFINE_INSTANCE(instance_type, -1, -1, -1, d_h, d_w, d_d) \
                SPARSE_CONV3D_DEFINE_INSTANCE(instance_type, 3, 3, 3, d_h, d_w, d_d)

#define SPARSE_CONV3D_DEFINE_INSTANCE_WITH_TYPE(instance_type) \
                      SPARSE_CONV3D_DEFINE_INSTANCE_WITH_DILATIONS(instance_type, -1, -1, -1)\
                      SPARSE_CONV3D_DEFINE_INSTANCE_WITH_DILATIONS(instance_type, 1, 1, 1) \
                      SPARSE_CONV3D_DEFINE_INSTANCE_WITH_DILATIONS(instance_type, 2, 2, 2)

SPARSE_CONV3D_DEFINE_INSTANCE_WITH_TYPE(float);
SPARSE_CONV3D_DEFINE_INSTANCE_WITH_TYPE(double); 

#undef SPARSE_CONV3D_DEFINE_INSTANCE

template <typename T, typename INDEX_TYPE, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
__global__ void SparseConv3DGradKernel(const int32 count,
              SPARSE_CONV3D_KERNEL_BASE_ARG_DEF_LIST,
              const T * out_grad_data,
              T* images_grad_data,
              T* filter_grad_data,
              T* default_channel_grad) {
  // GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(T), unsigned char , shared_memory);
  // T* const shared_data = reinterpret_cast<T*>(shared_memory);
  const int filter_h = kKnownFilterHeight < 0 ? filter_h_arg: kKnownFilterHeight;
  const int filter_w = kKnownFilterWidth < 0 ? filter_w_arg: kKnownFilterWidth;
  const int filter_d = kKnownFilterDepth < 0 ? filter_d_arg: kKnownFilterDepth;
  const int dilations_h = kKnownDilationHeight < 0 ? dilations_h_arg: kKnownDilationHeight;
  const int dilations_w = kKnownDilationWidth < 0 ? dilations_w_arg: kKnownDilationWidth;
  const int dilations_d = kKnownDilationDepth < 0 ? dilations_d_arg: kKnownDilationDepth;

  const INDEX_TYPE image_width_step = image_depth*image_channels;
  const INDEX_TYPE image_height_step = image_width*image_width_step;
  const INDEX_TYPE image_batch_step = image_height*image_height_step;
  const auto filter_depth_step = image_channels*out_channel_num;
  const auto filter_width_step = filter_d*filter_depth_step;
  const auto filter_height_step = filter_w*filter_width_step;
  for (const auto i : GpuGridRangeX<INDEX_TYPE>(count)) {
    const int32 d = i%image_depth;
    const INDEX_TYPE depth_map_pos = i/image_depth;
 
    const int32 w = depth_map_pos%image_width;
    const int32 tmp = depth_map_pos/image_width;

    const int32 h = tmp%image_height;
    const int32 b = tmp/image_height;

    const auto image_start_h = h - (filter_h/2)*dilations_h;
    const auto image_start_w = w - (filter_w/2)*dilations_w;
    const auto image_start_d = d - (filter_d/2)*dilations_d;

    const auto image_batch_ptr = &images_data[b*image_batch_step];
    const auto image_grad_batch_ptr = &images_grad_data[b*image_batch_step];
    const auto base_plane_batch_ptr = &base_plane_data[b*image_height*image_width];

    // auto out_channels = &out_data[i*out_channel_num];

    // for(int o_c = 0; o_c < out_channel_num; o_c++){
    //   out_channels[o_c] = 0.;
    // }
    const auto out_grad_channel = &out_grad_data[i*out_channel_num];

    T default_value_grad_tmp = T(0.);

    _Pragma("unroll")  for(int f_h = 0; f_h < filter_h; f_h++){
      const auto im_h = image_start_h + f_h*dilations_h;
      const auto f_h_ptr = &filter_data[f_h*filter_height_step];
      const auto f_grad_h_ptr = &filter_grad_data[f_h*filter_height_step];
      if((im_h >= 0) && (im_h < image_height)){
        /* 1. valid height pixel */
        const auto im_h_ptr = &image_batch_ptr[im_h*image_height_step];
        const auto im_grad_h_ptr = &image_grad_batch_ptr[im_h*image_height_step];
        const auto base_plane_h_ptr = &base_plane_batch_ptr[im_h*image_width];
        _Pragma("unroll")   for(int f_w = 0; f_w < filter_w; f_w++){
          const auto im_w = image_start_w + f_w*dilations_w;
          const auto f_w_ptr = &f_h_ptr[f_w*filter_width_step];
          const auto f_grad_w_ptr = &f_grad_h_ptr[f_w*filter_width_step];
          if((im_w >= 0) && (im_w < image_width)){
            /* 2. valid width pixel */
            const auto im_w_ptr = &im_h_ptr[im_w*image_width_step];
            const auto im_grad_w_ptr = &im_grad_h_ptr[im_w*image_width_step];
            const auto base_delta_d = image_start_d + base_plane_data[depth_map_pos] - base_plane_h_ptr[im_w];
            _Pragma("unroll")   for(int f_d = 0; f_d < filter_d; f_d++){
              const auto im_d = base_delta_d + f_d*dilations_d;
              const auto f_d_ptr = &f_w_ptr[f_d*filter_depth_step];
              const auto f_grad_d_ptr = &f_grad_w_ptr[f_d*filter_depth_step];
              if((im_d >= 0) && (im_d < image_depth)){
                /* 3. valid depth pixel */
                const auto im_d_ptr = &im_w_ptr[im_d*image_channels];
                const auto im_grad_d_ptr = &im_grad_w_ptr[im_d*image_channels];
                for(int f_c = 0; f_c < image_channels; f_c++){
                  const auto f_o_ptr = &f_d_ptr[f_c*out_channel_num];
                  const auto f_grad_o_ptr = &f_grad_d_ptr[f_c*out_channel_num];
                  T tmp = T(0);
                  for(int o_c = 0; o_c < out_channel_num; o_c++){
                    /* output channel loop */
                    atomicAdd(&f_grad_o_ptr[o_c], im_d_ptr[f_c]*out_grad_channel[o_c]);
                    tmp += f_o_ptr[o_c]*out_grad_channel[o_c];
                  }
                  atomicAdd(&im_grad_d_ptr[f_c], tmp);
                }
              } else {
                /* 3. empty depth pixel */
                for(int32 f_pos = 0; f_pos < filter_depth_step; f_pos = f_pos + out_channel_num){
                  const auto f_o_ptr = &f_d_ptr[f_pos];
                  const auto f_grad_o_ptr = &f_grad_d_ptr[f_pos];
                  for(int o_c = 0; o_c < out_channel_num; o_c++){
                    // out_channels[o_c] += default_channel_value[0]*f_o_ptr[o_c];
                    atomicAdd(&f_grad_o_ptr[o_c], default_channel_value[0]*out_grad_channel[o_c]);
                    default_value_grad_tmp += f_o_ptr[o_c]*out_grad_channel[o_c];
                  }
                }
              }
            }
          } else {
            /* 2. empty width pixel */
            for(int32 f_pos = 0; f_pos < filter_width_step; f_pos = f_pos + out_channel_num){
              const auto f_o_ptr = &f_w_ptr[f_pos];
              const auto f_grad_o_ptr = &f_grad_w_ptr[f_pos];
              for(int o_c = 0; o_c < out_channel_num; o_c++){
                // out_channels[o_c] += default_channel_value[0]*f_o_ptr[o_c];
                atomicAdd(&f_grad_o_ptr[o_c], default_channel_value[0]*out_grad_channel[o_c]);
                default_value_grad_tmp += f_o_ptr[o_c]*out_grad_channel[o_c];
              }
            }
          }
        }
      } else {
        /* 1. empty height pixel */
        for(int32 f_pos = 0; f_pos < filter_height_step; f_pos = f_pos + out_channel_num){
          const auto f_o_ptr = &f_h_ptr[f_pos];
          const auto f_grad_o_ptr = &f_grad_h_ptr[f_pos];
          for(int o_c = 0; o_c < out_channel_num; o_c++){
            // out_channels[o_c] += default_channel_value[0]*f_o_ptr[o_c];
            atomicAdd(&f_grad_o_ptr[o_c], default_channel_value[0]*out_grad_channel[o_c]);
            default_value_grad_tmp += f_o_ptr[o_c]*out_grad_channel[o_c];
          }
        }
      }
    }
    if(default_value_grad_tmp > T(0.)){
      atomicAdd(&default_channel_grad[0], default_value_grad_tmp);
    }
  }
}

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

#define SPARSE_CONV3D_KERNEL_GRAD_ARG_LIST \
              loop_count, \
              SPARSE_CONV3D_KERNEL_BASE_ARG_LIST,\
              out_grad_data, \
              images_grad_data, \
              filter_grad_data, \
              default_channel_value_grad 


template <typename T, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
void SparseConv3DGradFunctor<Eigen::GpuDevice, T, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>::operator()(const Eigen::GpuDevice& dev, 
              SPARSE_CONV3D_BASE_ARG_DEF_LIST,
              const T * out_grad_data,
              T * images_grad_data,
              T * filter_grad_data,
              T * default_channel_value_grad)
{
  const auto loop_count = static_cast<int64>(batch_size)*image_height*image_width*image_depth;
  const auto images_size = static_cast<int64>(batch_size)*image_height*image_width*image_depth*image_channels;
  const auto out_size = static_cast<int64>(batch_size)*image_height*image_width*image_depth*out_channel_num;
  const auto filter_size= filter_h*filter_w*filter_d*image_channels*out_channel_num;
  if( (images_size > INT32_MAX) | (out_size > INT32_MAX)){
    auto config = GetGpuLaunchConfigBig(images_size, dev, SetZeroBig<T, int64>, 0, 0);
    SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(images_size, images_grad_data);
    config = GetGpuLaunchConfigBig(filter_size, dev, SetZeroBig<T, int64>, 0, 0);
    SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(filter_size, filter_grad_data);
    config = GetGpuLaunchConfigBig(1, dev, SetZeroBig<T, int64>, 0, 0);
    SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(1, default_channel_value_grad);

    config = GetGpuLaunchConfig(loop_count, dev, SparseConv3DGradKernel<T, int64, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>, 0, 0);
    SparseConv3DGradKernel<T, int64, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(SPARSE_CONV3D_KERNEL_GRAD_ARG_LIST);  
  } else {
    auto config = GetGpuLaunchConfigBig(images_size, dev, SetZeroBig<T, int32>, 0, 0);
    SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(images_size, images_grad_data);
    config = GetGpuLaunchConfigBig(filter_size, dev, SetZeroBig<T, int32>, 0, 0);
    SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(filter_size, filter_grad_data);
    config = GetGpuLaunchConfigBig(1, dev, SetZeroBig<T, int32>, 0, 0);
    SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(1, default_channel_value_grad);

    config = GetGpuLaunchConfig(loop_count, dev, SparseConv3DGradKernel<T, int32, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST>, 0, 0);
    SparseConv3DGradKernel<T, int32, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(SPARSE_CONV3D_KERNEL_GRAD_ARG_LIST);  
  }
}


#define SPARSE_CONV3D_DEFINE_INSTANCE(instance_type, f_h, f_w, f_d, d_h, d_w, d_d) \
                template struct SparseConv3DGradFunctor<GPUDevice, instance_type, f_h, f_w, f_d, d_h, d_w, d_d>;


SPARSE_CONV3D_DEFINE_INSTANCE_WITH_TYPE(float);
SPARSE_CONV3D_DEFINE_INSTANCE_WITH_TYPE(double); 

#undef SPARSE_CONV3D_DEFINE_INSTANCE

} /* functor */
} /* custom_helper_op */
} /* tensorflow */

#endif /*EIGEN_USE_GPU */