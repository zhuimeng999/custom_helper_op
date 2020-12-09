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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "custom_helper_op/cc/kernels/sparse_pad.h"

namespace tensorflow {
namespace custom_helper_op {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

#define SPARSE_CONV3D_KERNEL_BASE_ARG_DEF_LIST \
              const int stride_h_arg, \
              const int stride_w_arg, \
              const int stride_d_arg, \
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
              const int out_height, \
              const int out_width, \
              const int out_depth, \
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
              out_height, \
              out_width, \
              out_depth, \
              out_channel_num, \
              images_data, \
              filter_data, \
              default_channel_value, \
              base_plane_data


// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always
// positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than
// 0x800...
// The casting allows to use one condition instead of two.
#define CHECK_PADDING(a, b) (static_cast<unsigned int>(a) >= static_cast<unsigned int>(b))

typedef Eigen::GpuDevice GPUDevice;

template <typename T, bool kCol2Im>
__global__ void Im2ColNdNCHWCUDAKernel(
    const SparseConv3DFastParams p,
    const int32 kernel_size,
    const int32 outer_size,
    const int32 inner_size,
    const int32* base_plane_data,
    const T* X_data,
    T* Y_data) {
  int d_offset_r, d_offset_c, d_offset_d;
  int d_iter_r, d_iter_c, d_iter_d;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    d_offset_d   =  i        % p.filter_depths;
    int offset_i =  i        / p.filter_depths;
    d_offset_c   =  offset_i % p.filter_cols;
    offset_i     =  offset_i / p.filter_cols;
    d_offset_r   =  offset_i % p.filter_rows;

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      d_iter_d     = j        % p.output_depths;
      int offset_j = j        / p.output_depths;
      d_iter_c     = offset_j % p.output_cols;
      offset_j     = offset_j / p.output_cols;
      d_iter_r     = offset_j % p.output_rows;

      const int col_index = i * inner_size + j;
      int img_index = i / kernel_size;
      bool is_padding = false;

      int d_img = d_iter_r * p.stride_rows - p.padding_rows + d_offset_r * p.dilation_rows;
      is_padding |= CHECK_PADDING(d_img, p.input_rows);
      img_index = img_index * p.input_rows + d_img;

      d_img = d_iter_c * p.stride_cols - p.padding_cols + d_offset_c * p.dilation_cols;
      is_padding |= CHECK_PADDING(d_img, p.input_cols);
      img_index = img_index * p.input_cols + d_img;

      d_img = d_iter_d * p.stride_depths - p.padding_depths + d_offset_d * p.dilation_depths;
      is_padding |= CHECK_PADDING(d_img, p.input_depths);
      img_index = img_index * p.input_depths + d_img;

      if (!kCol2Im) {
        Y_data[col_index] = is_padding ? 0 : __ldg(X_data + img_index);
      } else if (!is_padding) {
        atomicAdd(Y_data + img_index, __ldg(X_data + col_index));
      }
    }
  }
}


#define SPARSE_CONV3D_KERNEL_ARG_LIST loop_count, SPARSE_CONV3D_KERNEL_BASE_ARG_LIST, out_data

template <typename T>
void SparseConv3DFastFunctor<Eigen::GpuDevice, T>::operator()(const Eigen::GpuDevice& d, const SparseConv3DFastParams &p)
{

};
template struct SparseConv3DFastFunctor<Eigen::GpuDevice, float>;
template struct SparseConv3DFastFunctor<Eigen::GpuDevice, double>;


#undef SPARSE_CONV3D_DEFINE_INSTANCE

template <typename T, bool dynamic_default, typename INDEX_TYPE, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
__global__ void SparseConv3DFastGradKernel(const int32 count,
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
  const int stride_h = kKnownStrideHeight < 0 ? stride_h_arg: kKnownStrideHeight;
  const int stride_w = kKnownStrideWidth < 0 ? stride_w_arg: kKnownStrideWidth;
  const int stride_d = kKnownStrideDepth < 0 ? stride_d_arg: kKnownStrideDepth;

  const INDEX_TYPE image_width_step = image_depth*image_channels;
  const INDEX_TYPE image_height_step = image_width*image_width_step;
  const INDEX_TYPE image_batch_step = image_height*image_height_step;
  const auto filter_depth_step = image_channels*out_channel_num;
  const auto filter_width_step = filter_d*filter_depth_step;
  const auto filter_height_step = filter_w*filter_width_step;
  for (const auto i : GpuGridRangeX<INDEX_TYPE>(count)) {
    int32 d = i%out_depth;
    INDEX_TYPE depth_map_pos = i/out_depth;
 
    int32 w = depth_map_pos%out_width;
    const int32 tmp = depth_map_pos/out_width;

    int32 h = tmp%out_height;
    const int32 b = tmp/out_height;
    d = d*stride_d;
    w = w*stride_w;
    h = h*stride_h;

    if((kKnownStrideHeight != 1) || (kKnownStrideWidth != 1)){
      depth_map_pos = (b*image_height+h)*image_width+w;
    }

    d = d + ((base_plane_data[depth_map_pos] + stride_d - 1)/stride_d)*stride_d - base_plane_data[depth_map_pos];

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

    T default_value_grad_tmp;
    if(dynamic_default){
      default_value_grad_tmp = T(0.);
    }

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
                    GpuAtomicAdd(&f_grad_o_ptr[o_c], im_d_ptr[f_c]*out_grad_channel[o_c]);
                    tmp += f_o_ptr[o_c]*out_grad_channel[o_c];
                  }
                  GpuAtomicAdd(&im_grad_d_ptr[f_c], tmp);
                }
              } else {
                /* 3. empty depth pixel */
                for(int32 f_pos = 0; f_pos < filter_depth_step; f_pos = f_pos + out_channel_num){
                  const auto f_o_ptr = &f_d_ptr[f_pos];
                  const auto f_grad_o_ptr = &f_grad_d_ptr[f_pos];
                  for(int o_c = 0; o_c < out_channel_num; o_c++){
                    // out_channels[o_c] += default_channel_value[0]*f_o_ptr[o_c];
                    GpuAtomicAdd(&f_grad_o_ptr[o_c], default_channel_value[0]*out_grad_channel[o_c]);
                    if(dynamic_default){
                      default_value_grad_tmp += f_o_ptr[o_c]*out_grad_channel[o_c];
                    }
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
                GpuAtomicAdd(&f_grad_o_ptr[o_c], default_channel_value[0]*out_grad_channel[o_c]);
                if(dynamic_default){
                  default_value_grad_tmp += f_o_ptr[o_c]*out_grad_channel[o_c];
                }
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
            GpuAtomicAdd(&f_grad_o_ptr[o_c], default_channel_value[0]*out_grad_channel[o_c]);
            if(dynamic_default){
              default_value_grad_tmp += f_o_ptr[o_c]*out_grad_channel[o_c];
            }
          }
        }
      }
    }
    if(dynamic_default){
      if(default_value_grad_tmp > T(0.)){
        GpuAtomicAdd(&default_channel_grad[0], default_value_grad_tmp);
      }
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


template <typename T>
void SparseConv3DFastGradFunctor<Eigen::GpuDevice, T>::operator()(const Eigen::GpuDevice& dev, 
              SPARSE_CONV3D_BASE_ARG_DEF_LIST,
              const T * out_grad_data,
              T * images_grad_data,
              T * filter_grad_data,
              T * default_channel_value_grad)
{

}


}  // end namespace functor

}  // namespace addons
}  // namespace tensorflow

#endif  // GOOGLE_CUDA