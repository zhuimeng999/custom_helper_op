/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "custom_helper_op/cc/kernels/cost_volume_v3.h"

namespace tensorflow {
namespace custom_helper_op {

namespace functor {
// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename INDEX_TYPE, bool half_centor>
__global__ void CostMeanVolumeV3Kernel(const INDEX_TYPE virtual_thread, 
              const INDEX_TYPE batch_size, 
              const INDEX_TYPE image_height, 
              const INDEX_TYPE image_width,
              const INDEX_TYPE image_channels,
              const INDEX_TYPE image_depth,
              const INDEX_TYPE src_image_num,
              const INDEX_TYPE src_image_height, 
              const INDEX_TYPE src_image_width,
              const int32 groups,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              T* cost_data,
              int32* cost_mask_data){
  T src_channel_buffer[MAX_SUBCHANNELS_SIZE];

  const auto src_image_height_step = src_image_width * image_channels;
  int sub_channels = image_channels/groups;
  for (const auto i : GpuGridRangeX<INDEX_TYPE>(virtual_thread)){
    const auto batch_step = i/image_depth;
    const auto d = i%image_depth;

    const auto tmp = batch_step/image_width;
    const auto w = batch_step%image_width;
    const auto b = tmp/image_height;
    const auto h = tmp%image_height;

    const T * ref_channels = &ref_image_data[batch_step * image_channels];

    const T depth = base_plane_data[batch_step] + offsets_data[b*image_depth + d];
    T ref_w = static_cast<T>(w);
    T ref_h = static_cast<T>(h);
    if(half_centor){
      ref_w = ref_w + 0.5;
      ref_h = ref_h + 0.5;
    }

    int32 used_sample = 0;
    T cost = T(0.);
    for(INDEX_TYPE n = 0; n < src_image_num; n++){
      const T *R = &Rs_data[(b*src_image_num + n)*9];
      const T *t = &Ts_data[(b*src_image_num + n)*3];

      T src_z_coef = R[6] * ref_w + R[7] * ref_h + R[8];
      T src_z = src_z_coef * depth + t[2];
      if(src_z <= 0.0f){
        continue;
      }
      T src_w_coef = R[0] * ref_w + R[1] * ref_h + R[2];
      T src_w_3d = src_w_coef * depth + t[0];
      T src_w = src_w_3d/src_z;
      T src_h_coef = R[3] * ref_w + R[4] * ref_h + R[5];
      T src_h_3d = src_h_coef * depth + t[1];
      T src_h = src_h_3d/src_z;
      if(half_centor){
        src_w = src_w - 0.5;
        src_h = src_h - 0.5;
      }

      if (src_h > 0.0f && src_w > 0.0f &&
        src_h < static_cast<T>(src_image_height - 1) && src_w < static_cast<T>(src_image_width - 1)) {
        const INDEX_TYPE fh = static_cast<INDEX_TYPE>(src_h);
        const INDEX_TYPE fw = static_cast<INDEX_TYPE>(src_w);
        const T dh = src_h - fh;
        const T dw = src_w - fw;
        const T coef_ff = dh*dw;
        const T coef_fc = dh*(1 - dw);
        const T coef_cc = (1 - dh)*(1 - dw);
        const T coef_cf = (1 - dh)*dw;

        const T *src_channels_ff = &src_images_data[image_channels*(fw + src_image_width*(fh + src_image_height*(n + src_image_num*b)))];
        const T *src_channels_fc = &src_channels_ff[image_channels];
        const T *src_channels_cc = &src_channels_fc[src_image_height_step];
        const T *src_channels_cf = &src_channels_ff[src_image_height_step];

        T curr_cost = std::numeric_limits<T>::min();
        for(int gs = 0; gs < groups; gs++){
          for(int gsc = 0; gsc < sub_channels; gsc++){
            int sc = gs * sub_channels + gsc;
            src_channel_buffer[gsc] = coef_cc*src_channels_ff[sc] + coef_cf*src_channels_fc[sc] +
                                    coef_ff*src_channels_cc[sc] + coef_fc*src_channels_cf[sc];
          }

          for(int gi = 0; gi < groups; gi++){
            T tmp = T(0.);
            for(int gic = 0; gic < sub_channels; gic++){
              tmp  += src_channel_buffer[gic] * ref_channels[gi * sub_channels + gic];
            }
            if(curr_cost < tmp){
              curr_cost = tmp;
            }
          }
        }
        cost += curr_cost;

        used_sample = used_sample + 1;
      }
    }
    cost_mask_data[i] = used_sample;
    if(used_sample > 0){
      cost_data[i] = cost/static_cast<T>(used_sample);
    } else {
      cost_data[i] = 0;
    }
  }
}

template <typename T, typename INDEX_TYPE, bool half_centor>
__global__ void CostMinVolumeV3Kernel(const INDEX_TYPE virtual_thread, 
              const INDEX_TYPE batch_size, 
              const INDEX_TYPE image_height, 
              const INDEX_TYPE image_width,
              const INDEX_TYPE image_channels,
              const INDEX_TYPE image_depth,
              const INDEX_TYPE src_image_num,
              const INDEX_TYPE src_image_height, 
              const INDEX_TYPE src_image_width,
              const int groups,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              T* cost_data,
              int32* cost_mask_data){
  T src_channel_buffer[MAX_SUBCHANNELS_SIZE];

  const auto src_image_height_step = src_image_width * image_channels;
  int sub_channels = image_channels/groups;

  for (const auto i : GpuGridRangeX<INDEX_TYPE>(virtual_thread)){
    const auto batch_step = i/image_depth;
    const auto d = i%image_depth;

    const auto tmp = batch_step/image_width;
    const auto w = batch_step%image_width;
    const auto b = tmp/image_height;
    const auto h = tmp%image_height;

    const T * ref_channels = &ref_image_data[batch_step * image_channels];

    const T depth = base_plane_data[batch_step] + offsets_data[b*image_depth + d];
    T ref_w = static_cast<T>(w);
    T ref_h = static_cast<T>(h);
    if(half_centor){
      ref_w = ref_w + 0.5;
      ref_h = ref_h + 0.5;
    }

    auto cost_mask_info_ptr = &cost_mask_data[i*3];
    _Pragma("unroll") for(int m = 0; m < 3; m++){
      cost_mask_info_ptr[m] = -1;
    }

    T cost = std::numeric_limits<T>::min();

    for(INDEX_TYPE n = 0; n < src_image_num; n++){
      const T *R = &Rs_data[(b*src_image_num + n)*9];
      const T *t = &Ts_data[(b*src_image_num + n)*3];

      T src_z_coef = R[6] * ref_w + R[7] * ref_h + R[8];
      T src_z = src_z_coef * depth + t[2];
      if(src_z <= 0.0f){
        continue;
      }
      T src_w_coef = R[0] * ref_w + R[1] * ref_h + R[2];
      T src_w_3d = src_w_coef * depth + t[0];
      T src_w = src_w_3d/src_z;
      T src_h_coef = R[3] * ref_w + R[4] * ref_h + R[5];
      T src_h_3d = src_h_coef * depth + t[1];
      T src_h = src_h_3d/src_z;
      if(half_centor){
        src_w = src_w - 0.5;
        src_h = src_h - 0.5;
      }

      if (src_h > 0.0f && src_w > 0.0f &&
        src_h < static_cast<T>(src_image_height - 1) && src_w < static_cast<T>(src_image_width - 1)) {
        const INDEX_TYPE fh = static_cast<INDEX_TYPE>(src_h);
        const INDEX_TYPE fw = static_cast<INDEX_TYPE>(src_w);
        const T dh = src_h - fh;
        const T dw = src_w - fw;
        const T coef_ff = dh*dw;
        const T coef_fc = dh*(1 - dw);
        const T coef_cc = (1 - dh)*(1 - dw);
        const T coef_cf = (1 - dh)*dw;

        const T *src_channels_ff = &src_images_data[image_channels*(fw + src_image_width*(fh + src_image_height*(n + src_image_num*b)))];
        const T *src_channels_fc = &src_channels_ff[image_channels];
        const T *src_channels_cc = &src_channels_fc[src_image_height_step];
        const T *src_channels_cf = &src_channels_ff[src_image_height_step];

        for(int gs = 0; gs < groups; gs++){
          for(int gsc = 0; gsc < sub_channels; gsc++){
            int sc = gs * sub_channels + gsc;
            src_channel_buffer[gsc] = coef_cc*src_channels_ff[sc] + coef_cf*src_channels_fc[sc] +
                                    coef_ff*src_channels_cc[sc] + coef_fc*src_channels_cf[sc];
          }

          for(int gi = 0; gi < groups; gi++){
            T tmp = T(0.);
            for(int gic = 0; gic < sub_channels; gic++){
              tmp += src_channel_buffer[gic] * ref_channels[gi * sub_channels + gic];
            }
            if(cost < tmp){
              cost = tmp;
              cost_mask_info_ptr[0] = n;
              cost_mask_info_ptr[1] = gi;
              cost_mask_info_ptr[2] = gs;
            }
          }
        }
      }
    }
    if(cost_mask_info_ptr[0] >=0){
      cost_data[i] = cost;
    } else {
      cost_data[i] = 0;
    }
    
  }
}

// Calculate the GPU launch config we should use for a kernel launch. This
// variant takes the resource limits of func into account to maximize occupancy.
// REQUIRES: work_element_count > 0.
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

#define COST_ARG_LIST \
                                  loop_count, \
                                  batch_size,  \
                                  image_height, \
                                  image_width, \
                                  image_channels, \
                                  image_depth, \
                                  src_image_num, \
                                  src_image_height, \
                                  src_image_width, \
                                  groups,\
                                  ref_image_data, \
                                  src_images_data, \
                                  base_plane_data, \
                                  offsets_data, \
                                  Rs_data, \
                                  Ts_data

// Define the GPU implementation that launches the CUDA kernel.
template <typename T, bool half_centor>
void CostVolumeV3Functor<Eigen::GpuDevice, T, half_centor>::operator()(
    const GPUDevice& dev, COST_REDUCE_METHOD reduce_method,
              const int64 batch_size, 
              const int64 image_height, 
              const int64 image_width,
              const int64 image_channels,
              const int64 image_depth,
              const int64 src_image_num,
              const int64 src_image_height, 
              const int64 src_image_width,
              const int groups,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              T* mapped_feature_data,
              int32* mapped_mask_data
                                ) {
    const auto loop_count = batch_size * image_height * image_width * image_depth;
    const auto input_ref_size = batch_size * image_height * image_width * image_channels;
    const auto input_src_size = batch_size * src_image_num * src_image_height * src_image_width * image_channels;
    const auto output_size = batch_size * image_height * image_width * image_depth * 1;
    if((input_ref_size > INT32_MAX) || (input_src_size > INT32_MAX) || (output_size > INT32_MAX)){
      if(reduce_method == COST_REDUCE_MEAN){
        auto config = GetGpuLaunchConfigBig(loop_count, dev, CostMeanVolumeV3Kernel<T, int64, half_centor>, 0, 0);
        CostMeanVolumeV3Kernel<T, int64, half_centor><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                  COST_ARG_LIST, mapped_feature_data, mapped_mask_data);
      } else {
        auto config = GetGpuLaunchConfigBig(loop_count, dev, CostMinVolumeV3Kernel<T, int64, half_centor>, 0, 0);
        CostMinVolumeV3Kernel<T, int64, half_centor><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                  COST_ARG_LIST, mapped_feature_data, mapped_mask_data);
      }

    } else {
      if(reduce_method == COST_REDUCE_MEAN){
        auto config = GetGpuLaunchConfigBig(loop_count, dev, CostMeanVolumeV3Kernel<T, int32, half_centor>, 0, 0);
        CostMeanVolumeV3Kernel<T, int32, half_centor><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                  COST_ARG_LIST, mapped_feature_data, mapped_mask_data);
      } else {
        auto config = GetGpuLaunchConfigBig(loop_count, dev, CostMinVolumeV3Kernel<T, int32, half_centor>, 0, 0);
        CostMinVolumeV3Kernel<T, int32, half_centor><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                  COST_ARG_LIST, mapped_feature_data, mapped_mask_data);
      }
    }
}

template struct CostVolumeV3Functor<GPUDevice, float, true>;
template struct CostVolumeV3Functor<GPUDevice, float, false>;
template struct CostVolumeV3Functor<GPUDevice, double, true>;
template struct CostVolumeV3Functor<GPUDevice, double, false>;

template <typename T, typename INDEX_TYPE, bool half_centor>
__global__ void CostMeanVolumeGradV3Kernel(const INDEX_TYPE virtual_thread, 
              const INDEX_TYPE batch_size, 
              const INDEX_TYPE image_height, 
              const INDEX_TYPE image_width,
              const INDEX_TYPE image_channels,
              const INDEX_TYPE image_depth,
              const INDEX_TYPE src_image_num,
              const INDEX_TYPE src_image_height, 
              const INDEX_TYPE src_image_width,
              const int32 groups,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              const T* cost_grad_data,
              const int32* cost_mask_data,
              T* ref_image_grad_data,
              T* src_images_grad_data, 
              T* base_plane_grad_data
              ){
  T src_channel_buffer[MAX_SUBCHANNELS_SIZE];
  const auto src_image_height_step = src_image_width * image_channels;
  const int sub_channels = image_channels/groups;

  for (const auto i : GpuGridRangeX<INDEX_TYPE>(virtual_thread)){
    if(cost_mask_data[i] <= 0){
      continue;
    }
    const auto batch_step = i/image_depth;
    const auto d = i%image_depth;

    const auto tmp = batch_step/image_width;
    const auto w = batch_step%image_width;
    const auto b = tmp/image_height;
    const auto h = tmp%image_height;

    const T * ref_channels = &ref_image_data[batch_step * image_channels];
    T * ref_out_channels = &ref_image_grad_data[batch_step * image_channels];

    const T depth = base_plane_data[batch_step] + offsets_data[b*image_depth + d];
    T ref_w = static_cast<T>(w);
    T ref_h = static_cast<T>(h);
    if(half_centor){
      ref_w = ref_w + 0.5;
      ref_h = ref_h + 0.5;
    }

    const auto cost_grad = cost_grad_data[i]/static_cast<T>(cost_mask_data[i]);

    T depth_grad = T(0);
    for(INDEX_TYPE n = 0; n < src_image_num; n++){
      const T *R = &Rs_data[(b*src_image_num + n)*9];
      const T *t = &Ts_data[(b*src_image_num + n)*3];

      T src_z_coef = R[6] * ref_w + R[7] * ref_h + R[8];
      T src_z = src_z_coef * depth + t[2];
      if(src_z <= 0.0f){
        continue;
      }
      T src_w_coef = R[0] * ref_w + R[1] * ref_h + R[2];
      T src_w_3d = src_w_coef * depth + t[0];
      T src_w = src_w_3d/src_z;
      T src_h_coef = R[3] * ref_w + R[4] * ref_h + R[5];
      T src_h_3d = src_h_coef * depth + t[1];
      T src_h = src_h_3d/src_z;
      if(half_centor){
        src_w = src_w - 0.5;
        src_h = src_h - 0.5;
      }

      if (src_h > 0.0f && src_w > 0.0f &&
        src_h < static_cast<T>(src_image_height - 1) && src_w < static_cast<T>(src_image_width - 1)) {
        const INDEX_TYPE fh = static_cast<INDEX_TYPE>(src_h);
        const INDEX_TYPE fw = static_cast<INDEX_TYPE>(src_w);
        const T dh = src_h - fh;
        const T dw = src_w - fw;
        const T coef_ff = dh*dw;
        const T coef_fc = dh*(1 - dw);
        const T coef_cc = (1 - dh)*(1 - dw);
        const T coef_cf = (1 - dh)*dw;

        const T *src_channels_ff = &src_images_data[image_channels*(fw + src_image_width*(fh + src_image_height*(n + src_image_num*b)))];
        T *src_out_channels_ff = &src_images_grad_data[image_channels*(fw + src_image_width*(fh + src_image_height*(n + src_image_num*b)))];
        const T *src_channels_fc = &src_channels_ff[image_channels];
        T *src_out_channels_fc = &src_out_channels_ff[image_channels];
        const T *src_channels_cc = &src_channels_fc[src_image_height_step];
        T *src_out_channels_cc = &src_out_channels_fc[src_image_height_step];
        const T *src_channels_cf = &src_channels_ff[src_image_height_step];
        T *src_out_channels_cf = &src_out_channels_ff[src_image_height_step];

        int max_gi = -1;
        int max_gs = -1;
        T curr_cost = std::numeric_limits<T>::min();
        for(int gs = 0; gs < groups; gs++){
          for(int gsc = 0; gsc < sub_channels; gsc++){
            int sc = gs * sub_channels + gsc;
            src_channel_buffer[gsc] = coef_cc*src_channels_ff[sc] + coef_cf*src_channels_fc[sc] +
                                    coef_ff*src_channels_cc[sc] + coef_fc*src_channels_cf[sc];
          }

          for(int gi = 0; gi < groups; gi++){
            T tmp = T(0.);
            for(int gic = 0; gic < sub_channels; gic++){
              tmp  += src_channel_buffer[gic] * ref_channels[gi * sub_channels + gic];
            }
            if(curr_cost < tmp){
              curr_cost = tmp;
              max_gi = gi;
              max_gs = gs;
            }
          }
        }

        T h_grad = T(0);
        T w_grad = T(0);
        for(int sc = 0; sc < sub_channels; sc++){
          int rc = max_gi * sub_channels + sc;
          int cc = max_gs * sub_channels + sc;
          T src_sample = coef_cc*src_channels_ff[cc] + coef_cf*src_channels_fc[cc] +
                                  coef_ff*src_channels_cc[cc] + coef_fc*src_channels_cf[cc];

          GpuAtomicAdd(&ref_out_channels[rc], cost_grad*src_sample);
          T base_grad = cost_grad*ref_channels[rc];
          GpuAtomicAdd(&src_out_channels_ff[cc], base_grad*coef_cc);
          GpuAtomicAdd(&src_out_channels_fc[cc], base_grad*coef_cf);
          GpuAtomicAdd(&src_out_channels_cc[cc], base_grad*coef_ff);
          GpuAtomicAdd(&src_out_channels_cf[cc], base_grad*coef_fc);

          // Update partial gradients wrt relevant warp field entries
          const auto t1 = src_channels_cc[cc]  - src_channels_cf[cc] - src_channels_fc[cc] + src_channels_ff[cc];
          h_grad += base_grad * (t1 * dw + src_channels_cf[cc] - src_channels_ff[cc]);
          w_grad += base_grad * (t1 * dh + src_channels_fc[cc] - src_channels_ff[cc]);
        }
        depth_grad += ((src_h_coef*src_z - src_z_coef*src_h_3d)*h_grad + (src_w_coef*src_z - src_z_coef*src_w_3d)*w_grad)/(src_z * src_z);
      }
    }
    GpuAtomicAdd(&base_plane_grad_data[batch_step], depth_grad);
  }
}

template <typename T, typename INDEX_TYPE, bool half_centor>
__global__ void CostMinVolumeGradV3Kernel(const INDEX_TYPE virtual_thread, 
              const INDEX_TYPE batch_size, 
              const INDEX_TYPE image_height, 
              const INDEX_TYPE image_width,
              const INDEX_TYPE image_channels,
              const INDEX_TYPE image_depth,
              const INDEX_TYPE src_image_num,
              const INDEX_TYPE src_image_height, 
              const INDEX_TYPE src_image_width,
              const int32 groups,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              const T* cost_grad_data,
              const int32* cost_mask_data,
              T* ref_image_grad_data,
              T* src_images_grad_data, 
              T* base_plane_grad_data
              ){
  const auto src_image_height_step = src_image_width * image_channels;
  const int sub_channels = image_channels/groups;

  for (const auto i : GpuGridRangeX<INDEX_TYPE>(virtual_thread)){
    const auto cost_mask_info_ptr = &cost_mask_data[i*3];
    if(cost_mask_info_ptr[0] < 0){
      continue;
    }

    const auto batch_step = i/image_depth;
    const auto d = i%image_depth;

    const auto tmp = batch_step/image_width;
    const auto w = batch_step%image_width;
    const auto b = tmp/image_height;
    const auto h = tmp%image_height;

    const T * ref_channels = &ref_image_data[batch_step * image_channels];
    T * ref_out_channels = &ref_image_grad_data[batch_step * image_channels];

    const T depth = base_plane_data[batch_step] + offsets_data[b*image_depth + d];
    T ref_w = static_cast<T>(w);
    T ref_h = static_cast<T>(h);
    if(half_centor){
      ref_w = ref_w + 0.5;
      ref_h = ref_h + 0.5;
    }

    T depth_grad = T(0);
    // for(INDEX_TYPE n = 0; n < src_image_num; n++){
    {
      INDEX_TYPE n = cost_mask_info_ptr[0];
      const T *R = &Rs_data[(b*src_image_num + n)*9];
      const T *t = &Ts_data[(b*src_image_num + n)*3];

      T src_z_coef = R[6] * ref_w + R[7] * ref_h + R[8];
      T src_z = src_z_coef * depth + t[2];
      if(src_z <= 0.0f){
        continue;
      }
      T src_w_coef = R[0] * ref_w + R[1] * ref_h + R[2];
      T src_w_3d = src_w_coef * depth + t[0];
      T src_w = src_w_3d/src_z;
      T src_h_coef = R[3] * ref_w + R[4] * ref_h + R[5];
      T src_h_3d = src_h_coef * depth + t[1];
      T src_h = src_h_3d/src_z;
      if(half_centor){
        src_w = src_w - 0.5;
        src_h = src_h - 0.5;
      }

      // if (src_h > 0.0f && src_w > 0.0f &&
      //   src_h < static_cast<T>(src_image_height - 1) && src_w < static_cast<T>(src_image_width - 1)) {
      {
        const INDEX_TYPE fh = static_cast<INDEX_TYPE>(src_h);
        const INDEX_TYPE fw = static_cast<INDEX_TYPE>(src_w);
        const T dh = src_h - fh;
        const T dw = src_w - fw;
        const T coef_ff = dh*dw;
        const T coef_fc = dh*(1 - dw);
        const T coef_cc = (1 - dh)*(1 - dw);
        const T coef_cf = (1 - dh)*dw;

        const T *src_channels_ff = &src_images_data[image_channels*(fw + src_image_width*(fh + src_image_height*(n + src_image_num*b)))];
        T *src_out_channels_ff = &src_images_grad_data[image_channels*(fw + src_image_width*(fh + src_image_height*(n + src_image_num*b)))];
        const T *src_channels_fc = &src_channels_ff[image_channels];
        T *src_out_channels_fc = &src_out_channels_ff[image_channels];
        const T *src_channels_cc = &src_channels_fc[src_image_height_step];
        T *src_out_channels_cc = &src_out_channels_fc[src_image_height_step];
        const T *src_channels_cf = &src_channels_ff[src_image_height_step];
        T *src_out_channels_cf = &src_out_channels_ff[src_image_height_step];

        T h_grad = T(0);
        T w_grad = T(0);
        for(int sc = 0; sc < sub_channels; sc++){
          int rc = cost_mask_info_ptr[1] * sub_channels + sc;
          int cc = cost_mask_info_ptr[2] * sub_channels + sc;
          T src_sample = coef_cc*src_channels_ff[cc] + coef_cf*src_channels_fc[cc] +
                                  coef_ff*src_channels_cc[cc] + coef_fc*src_channels_cf[cc];

          GpuAtomicAdd(&ref_out_channels[rc], cost_grad_data[i]*src_sample);
          T base_grad = cost_grad_data[i]*ref_channels[rc];
          GpuAtomicAdd(&src_out_channels_ff[cc], base_grad*coef_cc);
          GpuAtomicAdd(&src_out_channels_fc[cc], base_grad*coef_cf);
          GpuAtomicAdd(&src_out_channels_cc[cc], base_grad*coef_ff);
          GpuAtomicAdd(&src_out_channels_cf[cc], base_grad*coef_fc);

          // Update partial gradients wrt relevant warp field entries
          const auto t1 = src_channels_cc[cc]  - src_channels_cf[cc] - src_channels_fc[cc] + src_channels_ff[cc];
          h_grad += base_grad * (t1 * dw + src_channels_cf[cc] - src_channels_ff[cc]);
          w_grad += base_grad * (t1 * dh + src_channels_fc[cc] - src_channels_ff[cc]);
        }
        depth_grad += ((src_h_coef*src_z - src_z_coef*src_h_3d)*h_grad + (src_w_coef*src_z - src_z_coef*src_w_3d)*w_grad)/(src_z * src_z);
      }
    }
    GpuAtomicAdd(&base_plane_grad_data[batch_step], depth_grad);
  }
}

#define COST_GRAG_ARG_LIST COST_ARG_LIST, mapped_feature_grad_data, mapped_mask_data, ref_image_grad_data, src_images_grad_data, base_plane_grad_data
// Zeroes count elements starting at ptr using all threads of a 1-D grid.
// Note: this function does not synchronize, and therefore the memory range is
// not guaranteed to be zero until the next kernel launch.
template <typename T, typename INDEX_TYPE>
__global__ void SetZeroBig(const INDEX_TYPE count, T* __restrict__ ptr) {
  for (const auto i : GpuGridRangeX<INDEX_TYPE>(count)) {
    ptr[i] = T(0);
  }
}
// Define the GPU implementation that launches the CUDA kernel.
template <typename T, bool half_centor>
void CostVolumeGradV3Functor<Eigen::GpuDevice, T, half_centor>::operator()(
    const GPUDevice& dev, COST_REDUCE_METHOD reduce_method, 
              const int64 batch_size, 
              const int64 image_height, 
              const int64 image_width,
              const int64 image_channels,
              const int64 image_depth,
              const int64 src_image_num,
              const int64 src_image_height, 
              const int64 src_image_width,
              const int32 groups,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              const T* mapped_feature_grad_data,
              const int32* mapped_mask_data,
              T* ref_image_grad_data,
              T* src_images_grad_data, 
              T* base_plane_grad_data
                                ) {
    const auto base_plane_size = batch_size * image_height *image_width;
    const auto loop_count = batch_size * image_height * image_width * image_depth;
    const auto input_ref_size = batch_size * image_height * image_width * image_channels;
    const auto input_src_size = batch_size * src_image_num * src_image_height * src_image_width * image_channels;
    const auto output_size = batch_size * image_height * image_width * image_depth * 1;

    if((input_ref_size > INT32_MAX) || (input_src_size > INT32_MAX) || (output_size > INT32_MAX)){
      auto config = GetGpuLaunchConfigBig(input_ref_size, dev, SetZeroBig<T, int64>, 0, 0);
      SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(input_ref_size, ref_image_grad_data);
      config = GetGpuLaunchConfigBig(input_src_size, dev, SetZeroBig<T, int64>, 0, 0);
      SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(input_src_size, src_images_grad_data);
      config = GetGpuLaunchConfigBig(base_plane_size, dev, SetZeroBig<T, int64>, 0, 0);
      SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(base_plane_size, base_plane_grad_data);
      if(reduce_method == COST_REDUCE_MEAN){
        config = GetGpuLaunchConfigBig(loop_count, dev, CostMeanVolumeGradV3Kernel<T, int64, half_centor>, 0, 0);
        CostMeanVolumeGradV3Kernel<T, int64, half_centor><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                  COST_GRAG_ARG_LIST);
      } else {
        config = GetGpuLaunchConfigBig(loop_count, dev, CostMinVolumeGradV3Kernel<T, int64, half_centor>, 0, 0);
        CostMinVolumeGradV3Kernel<T, int64, half_centor><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                  COST_GRAG_ARG_LIST);
      }

    } else {
      auto config = GetGpuLaunchConfigBig(input_ref_size, dev, SetZeroBig<T, int32>, 0, 0);
      SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(input_ref_size, ref_image_grad_data);
      config = GetGpuLaunchConfigBig(input_src_size, dev, SetZeroBig<T, int32>, 0, 0);
      SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(input_src_size, src_images_grad_data);
      config = GetGpuLaunchConfigBig(base_plane_size, dev, SetZeroBig<T, int32>, 0, 0);
      SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(base_plane_size, base_plane_grad_data);

      if(reduce_method == COST_REDUCE_MEAN){
        config = GetGpuLaunchConfigBig(loop_count, dev, CostMeanVolumeGradV3Kernel<T, int32, half_centor>, 0, 0);
        CostMeanVolumeGradV3Kernel<T, int32, half_centor><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                  COST_GRAG_ARG_LIST);
      } else {
        config = GetGpuLaunchConfigBig(loop_count, dev, CostMinVolumeGradV3Kernel<T, int32, half_centor>, 0, 0);
        CostMinVolumeGradV3Kernel<T, int32, half_centor><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                  COST_GRAG_ARG_LIST);
      }

    }
}

template struct CostVolumeGradV3Functor<GPUDevice, float, true>;
template struct CostVolumeGradV3Functor<GPUDevice, float, false>;
template struct CostVolumeGradV3Functor<GPUDevice, double, true>;
template struct CostVolumeGradV3Functor<GPUDevice, double, false>;
}  // end namespace functor

}  // end namespace addons
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
