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
#include "custom_helper_op/cc/kernels/cost_aggregate.h"

namespace tensorflow {
namespace custom_helper_op {

namespace functor {
// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename INDEX_TYPE, COST_REDUCE_METHOD reduce_method>
__global__ void CostAggregateKernel(const INDEX_TYPE virtual_thread, 
              const INDEX_TYPE batch_size, 
              const INDEX_TYPE image_height, 
              const INDEX_TYPE image_width,
              const INDEX_TYPE image_channels,
              const INDEX_TYPE image_depth,
              const INDEX_TYPE src_image_num,
              const INDEX_TYPE src_image_height, 
              const INDEX_TYPE src_image_width,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              T* cost_data,
              int32* cost_mask_data){
  const auto src_image_height_step = src_image_width * image_channels;
  
  for (const auto i : GpuGridRangeX<INDEX_TYPE>(virtual_thread)){
    const auto batch_step = i/image_depth;
    const auto d = i%image_depth;

    const auto tmp = batch_step/image_width;
    const auto w = batch_step%image_width;
    const auto b = tmp/image_height;
    const auto h = tmp%image_height;

    const T * ref_channels = &ref_image_data[batch_step * image_channels];

    const T depth = base_plane_data[batch_step] + offsets_data[b*image_depth + d];
    const T ref_w = w*depth;
    const T ref_h = h*depth;

    T cost = T(0.);
    int32 used_sample = (reduce_method == COST_REDUCE_MEAN?0:-1);
    for(INDEX_TYPE n = 0; n < src_image_num; n++){
      const T *R = &Rs_data[(b*src_image_num + n)*9];
      const T *t = &Ts_data[(b*src_image_num + n)*3];

      T src_z = R[6] * ref_w + R[7] * ref_h + R[8] * depth + t[2];
      if(src_z <= 0.0f){
        continue;
      }
      T src_w = (R[0] * ref_w + R[1] * ref_h + R[2] * depth + t[0])/src_z;
      T src_h = (R[3] * ref_w + R[4] * ref_h + R[5] * depth + t[1])/src_z;

      if (src_h >= 0.0f && src_w >= 0.0f &&
        src_h <= static_cast<T>(src_image_height - 1) && src_w <= static_cast<T>(src_image_width - 1)) {
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

        T curr_cost = T(0.);
        for(int cc = 0; cc < image_channels; cc++){
          T src_sample = coef_cc*src_channels_ff[cc] + coef_cf*src_channels_fc[cc] +
                                  coef_ff*src_channels_cc[cc] + coef_fc*src_channels_cf[cc];
          T diff = src_sample - ref_channels[cc];
          curr_cost += diff*diff;
        }

        if(reduce_method == COST_REDUCE_MEAN){
          cost += curr_cost;
          used_sample = used_sample + 1;
        } else {
          if(cost > curr_cost){
            cost = curr_cost;
            used_sample = n;
          }
        }
      }
    }
    cost_mask_data[i] = used_sample;
    if(used_sample > 0){
      if(reduce_method == COST_REDUCE_MEAN){
        cost_data[i] = cost/static_cast<T>(used_sample*image_channels);
      } else {
        cost_data[i] = cost/static_cast<T>(image_channels);
      }
    } else {
      cost_data[i] = T(0);
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

// Define the GPU implementation that launches the CUDA kernel.
template <typename T, COST_REDUCE_METHOD reduce_method>
void CostAggregateFunctor<Eigen::GpuDevice, T, reduce_method>::operator()(
    const GPUDevice& dev, 
              const int64 batch_size, 
              const int64 image_height, 
              const int64 image_width,
              const int64 image_channels,
              const int64 image_depth,
              const int64 src_image_num,
              const int64 src_image_height, 
              const int64 src_image_width,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              T* cost_data,
              int32* cost_mask_data
                                ) {
    const auto loop_count = batch_size * image_height * image_width * image_depth;
    const auto input_ref_size = batch_size * image_height * image_width * image_channels;
    const auto input_src_size = batch_size * src_image_num * src_image_height * src_image_width * image_channels;
    if((input_ref_size > INT32_MAX) || (input_src_size > INT32_MAX) || (loop_count > INT32_MAX)){
      auto config = GetGpuLaunchConfigBig(loop_count, dev, CostAggregateKernel<T, int64, reduce_method>, 0, 0);
      CostAggregateKernel<T, int64, reduce_method><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                loop_count,
                                batch_size, 
                                image_height, 
                                image_width,
                                image_channels,
                                image_depth,
                                src_image_num,
                                src_image_height, 
                                src_image_width,
                                ref_image_data,
                                src_images_data, 
                                base_plane_data,
                                offsets_data,
                                Rs_data,
                                Ts_data,
                                cost_data,
                                cost_mask_data);
    } else {
      auto config = GetGpuLaunchConfigBig(loop_count, dev, CostAggregateKernel<T, int32, reduce_method>, 0, 0);
      CostAggregateKernel<T, int32, reduce_method><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                loop_count,
                                batch_size, 
                                image_height, 
                                image_width,
                                image_channels,
                                image_depth,
                                src_image_num,
                                src_image_height, 
                                src_image_width,
                                ref_image_data,
                                src_images_data, 
                                base_plane_data,
                                offsets_data,
                                Rs_data,
                                Ts_data,
                                cost_data,
                                cost_mask_data);
    }
}

template struct CostAggregateFunctor<GPUDevice, float, COST_REDUCE_MEAN>;
template struct CostAggregateFunctor<GPUDevice, float, COST_REDUCE_MIN>;
template struct CostAggregateFunctor<GPUDevice, double, COST_REDUCE_MEAN>;
template struct CostAggregateFunctor<GPUDevice, double, COST_REDUCE_MIN>;

template <typename T, typename INDEX_TYPE, COST_REDUCE_METHOD reduce_method>
__global__ void CostAggregateGradKernel(const INDEX_TYPE virtual_thread, 
              const INDEX_TYPE batch_size, 
              const INDEX_TYPE image_height, 
              const INDEX_TYPE image_width,
              const INDEX_TYPE image_channels,
              const INDEX_TYPE image_depth,
              const INDEX_TYPE src_image_num,
              const INDEX_TYPE src_image_height, 
              const INDEX_TYPE src_image_width,
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

    T cost_grad = cost_grad_data[i]/static_cast<T>(cost_mask_data[i]*static_cast<int32>(image_channels));

    T depth_grad = T(0);
    for(INDEX_TYPE n = (reduce_method == COST_REDUCE_MEAN?0:cost_mask_data[i]); n < (reduce_method == COST_REDUCE_MEAN?src_image_num:(cost_mask_data[i] + 1)); n++){
      const T *R = &Rs_data[(b*src_image_num + n)*9];
      const T *t = &Ts_data[(b*src_image_num + n)*3];

      T src_z_coef = R[6] * w + R[7] * h + R[8];
      T src_z = src_z_coef * depth + t[2];
      if(src_z <= 0.0f){
        continue;
      }
      T src_w_coef = R[0] * w + R[1] * h + R[2];
      T src_w_3d = src_w_coef * depth + t[0];
      T src_w = src_w_3d/src_z;
      T src_h_coef = R[3] * w + R[4] * h + R[5];
      T src_h_3d = src_h_coef * depth + t[1];
      T src_h = src_h_3d/src_z;

      if (src_h >= 0.0f && src_w >= 0.0f &&
        src_h <= static_cast<T>(src_image_height - 1) && src_w <= static_cast<T>(src_image_width - 1)) {
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
        for(int cc = 0; cc < image_channels; cc++){
          T src_sample = coef_cc*src_channels_ff[cc] + coef_cf*src_channels_fc[cc] +
                                  coef_ff*src_channels_cc[cc] + coef_fc*src_channels_cf[cc];
          T diff = src_sample - ref_channels[cc];

          T ref_grad = 2*diff*cost_grad;
          GpuAtomicAdd(&ref_out_channels[cc], -ref_grad);
          GpuAtomicAdd(&src_out_channels_ff[cc], ref_grad*coef_cc);
          GpuAtomicAdd(&src_out_channels_fc[cc], ref_grad*coef_cf);
          GpuAtomicAdd(&src_out_channels_cc[cc], ref_grad*coef_ff);
          GpuAtomicAdd(&src_out_channels_cf[cc], ref_grad*coef_fc);

          // Update partial gradients wrt relevant warp field entries
          const auto t1 = src_channels_ff[cc] - src_channels_fc[cc];
          const auto t2 = src_channels_cc[cc]  - src_channels_cf[cc] + t1;
          h_grad += ref_grad * (t2 * dw - t1);
          w_grad += ref_grad * (t2 * dh - t1);
        }
        depth_grad += ((src_h_coef*src_z - src_z_coef*src_h_3d)*h_grad + (src_w_coef*src_z - src_z_coef*src_w_3d)*w_grad)/(src_z * src_z);
      }
    }
    GpuAtomicAdd(&base_plane_grad_data[batch_step], depth_grad);
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
// Define the GPU implementation that launches the CUDA kernel.
template <typename T, COST_REDUCE_METHOD reduce_method>
void CostAggregateGradFunctor<Eigen::GpuDevice, T, reduce_method>::operator()(
    const GPUDevice& dev, 
              const int64 batch_size, 
              const int64 image_height, 
              const int64 image_width,
              const int64 image_channels,
              const int64 image_depth,
              const int64 src_image_num,
              const int64 src_image_height, 
              const int64 src_image_width,
              const T* ref_image_data,
              const T* src_images_data, 
              const T* base_plane_data,
              const T* offsets_data,
              const T* Rs_data,
              const T* Ts_data,
              const T* cost_data,
              const int32* cost_mask_data,
              T* ref_image_grad_data,
              T* src_images_grad_data, 
              T* base_plane_grad_data
                                ) {
    const auto base_plane_size = batch_size * image_height *image_width;
    const auto loop_count = base_plane_size * image_depth;
    const auto input_ref_size = base_plane_size * image_channels;
    const auto input_src_size = batch_size * src_image_num * src_image_height * src_image_width * image_channels;

    if((input_ref_size > INT32_MAX) || (input_src_size > INT32_MAX) || (loop_count > INT32_MAX)){
      auto config = GetGpuLaunchConfigBig(input_ref_size, dev, SetZeroBig<T, int64>, 0, 0);
      SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(input_ref_size, ref_image_grad_data);
      config = GetGpuLaunchConfigBig(input_src_size, dev, SetZeroBig<T, int64>, 0, 0);
      SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(input_src_size, src_images_grad_data);
      config = GetGpuLaunchConfigBig(base_plane_size, dev, SetZeroBig<T, int64>, 0, 0);
      SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(base_plane_size, base_plane_grad_data);

      config = GetGpuLaunchConfigBig(loop_count, dev, CostAggregateGradKernel<T, int64, reduce_method>, 0, 0);
      CostAggregateGradKernel<T, int64, reduce_method><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                loop_count,
                                batch_size, 
                                image_height, 
                                image_width,
                                image_channels,
                                image_depth,
                                src_image_num,
                                src_image_height, 
                                src_image_width,
                                ref_image_data,
                                src_images_data, 
                                base_plane_data,
                                offsets_data,
                                Rs_data,
                                Ts_data,
                                cost_data,
                                cost_mask_data,
                                ref_image_grad_data,
                                src_images_grad_data, 
                                base_plane_grad_data);
    } else {
      auto config = GetGpuLaunchConfigBig(input_ref_size, dev, SetZeroBig<T, int32>, 0, 0);
      SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(input_ref_size, ref_image_grad_data);
      config = GetGpuLaunchConfigBig(input_src_size, dev, SetZeroBig<T, int32>, 0, 0);
      SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(input_src_size, src_images_grad_data);
      config = GetGpuLaunchConfigBig(base_plane_size, dev, SetZeroBig<T, int32>, 0, 0);
      SetZeroBig<T, int32><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(base_plane_size, base_plane_grad_data);

      config = GetGpuLaunchConfigBig(loop_count, dev, CostAggregateGradKernel<T, int32, reduce_method>, 0, 0);
      CostAggregateGradKernel<T, int32, reduce_method><<<config.block_count, config.thread_per_block, 0, dev.stream()>>>(
                                loop_count,
                                batch_size, 
                                image_height, 
                                image_width,
                                image_channels,
                                image_depth,
                                src_image_num,
                                src_image_height, 
                                src_image_width,
                                ref_image_data,
                                src_images_data, 
                                base_plane_data,
                                offsets_data,
                                Rs_data,
                                Ts_data,
                                cost_data,
                                cost_mask_data,
                                ref_image_grad_data,
                                src_images_grad_data, 
                                base_plane_grad_data);
    }
}

template struct CostAggregateGradFunctor<GPUDevice, float, COST_REDUCE_MEAN>;
template struct CostAggregateGradFunctor<GPUDevice, float, COST_REDUCE_MIN>;
template struct CostAggregateGradFunctor<GPUDevice, double, COST_REDUCE_MEAN>;
template struct CostAggregateGradFunctor<GPUDevice, double, COST_REDUCE_MIN>;

}  // end namespace functor

}  // end namespace addons
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
