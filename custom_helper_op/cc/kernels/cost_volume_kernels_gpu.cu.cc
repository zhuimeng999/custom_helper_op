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
#include "custom_helper_op/cc/kernels/cost_volume.h"

namespace tensorflow {
namespace custom_helper_op {

namespace functor {

// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename INDEX_TYPE, Interpolation INTERPOLATION_TYPE>
__global__ void CostVolumeKernel(const INDEX_TYPE virtual_thread, 
              const INDEX_TYPE batch_size, const INDEX_TYPE image_height, const INDEX_TYPE image_width, 
              const INDEX_TYPE image_depth, const INDEX_TYPE image_channels, const INDEX_TYPE image_num,
              const T* images_data, const T* transforms_data, T* out_data, T* out_mask_data){
  const INDEX_TYPE img_height_step = image_width*image_channels;
  const INDEX_TYPE img_step =  image_height*img_height_step;
  const INDEX_TYPE batch_img_step = image_num*img_step;
  const INDEX_TYPE homos_step = image_depth*8;
  const INDEX_TYPE batch_homos_step = (image_num - 1)*homos_step;

  for (const auto i : GpuGridRangeX<INDEX_TYPE>(virtual_thread)){
    auto tmp = i/image_depth;
    const auto cd = i - tmp*image_depth;
    auto tmp1 = tmp;
    tmp = tmp/image_width;
    const auto cw = tmp1 - tmp*image_width;
    const auto cb = tmp/image_height;
    const auto ch = tmp - cb*image_height;

    const T* batch_img_ptr = &images_data[cb*batch_img_step];
    const T *ref_channels = &batch_img_ptr[(ch*image_width+cw)*image_channels];
    const T * curr_homos = &transforms_data[cb*batch_homos_step + cd*8];

    T *out_channels = &out_data[i*image_channels];
    for(INDEX_TYPE cc = 0; cc < image_channels; cc++){
      out_channels[cc] = T(0);
    }
    T used_sample = T(0);
    for(INDEX_TYPE cn = 1; cn < image_num; cn++){
      const T *src_homos = &curr_homos[(cn - 1)*homos_step];
      T projection = src_homos[6] * cw + src_homos[7] * ch + 1.f;
      if(projection == 0.0f){
        continue;
      }
      const T src_w = (src_homos[0] * cw + src_homos[1] * ch + src_homos[2]) / projection;
      const T src_h = (src_homos[3] * cw + src_homos[4] * ch + src_homos[5]) / projection;

      if (src_h > 0.0f && src_w > 0.0f &&
        src_h < static_cast<T>(image_height - 1) && src_w < static_cast<T>(image_width - 1)) {
        const T fh = std::floor(src_h);
        const T fw = std::floor(src_w);
        const T dh = src_h - fh;
        const T dw = src_w - fw;
        const T coef_ff = dh*dw;
        const T coef_fc = dh*(1 - dw);
        const T coef_cc = (1 - dh)*(1 - dw);
        const T coef_cf = (1 - dh)*dw;

        const T *src_channels_ff = &batch_img_ptr[((cn*image_height + static_cast<INDEX_TYPE>(fh))*image_width+static_cast<INDEX_TYPE>(fw))*image_channels];
        const T *src_channels_fc = &src_channels_ff[image_channels];
        const T *src_channels_cc = &src_channels_fc[img_height_step];
        const T *src_channels_cf = &src_channels_cc[image_channels];

        for(int cc = 0; cc < image_channels; cc++){
          T src_sample = coef_cc*src_channels_ff[cc] + coef_cf*src_channels_fc[cc] +
                                  coef_ff*src_channels_cc[cc] + coef_fc*src_channels_cf[cc];
          T diff = src_sample - ref_channels[cc];
          out_channels[cc] += diff*diff;
        }
        used_sample = used_sample + 1;
      }
    }
    out_mask_data[i] = used_sample;
    if(used_sample > 0){
      for(int cc = 0; cc < image_channels; cc++){
        out_channels[cc] = out_channels[cc]/used_sample;
      }

    } else {
      for(int cc = 0; cc < image_channels; cc++){
        out_channels[cc] = 100;
      }
      out_mask_data[i] = T(0);
    }
  }
}

template <typename T, typename INDEX_TYPE, Interpolation INTERPOLATION_TYPE>
__global__ void CostVolumeKernelNoBatch(const INDEX_TYPE virtual_thread, 
              const INDEX_TYPE batch_size, const INDEX_TYPE image_height, const INDEX_TYPE image_width, 
              const INDEX_TYPE image_depth, const INDEX_TYPE image_channels, const INDEX_TYPE image_num,
              const T* images_data, const T* transforms_data, T* out_data,T* out_mask_data){
  (void )batch_size;
  const INDEX_TYPE img_height_step = image_width*image_channels;
  const INDEX_TYPE homos_step = image_depth*8;
  for (auto i : GpuGridRangeX<INDEX_TYPE>(virtual_thread)){
    auto tmp = i/image_depth;
    const auto cd = i - tmp*image_depth;
    const auto ch = tmp/image_width;
    const auto cw = tmp - ch*image_width;

    const T *ref_channels = &images_data[(ch*image_width+cw)*image_channels];
    const T *  curr_homos = &transforms_data[cd*8];

    T *out_channels = &out_data[i*image_channels];
    for(INDEX_TYPE cc = 0; cc < image_channels; cc++){
      out_channels[cc] = T(0);
    }
    T used_sample = T(0);
    for(INDEX_TYPE cn = 1; cn < image_num; cn++){
      const T *src_homos = &curr_homos[(cn - 1)*homos_step];
      T projection = src_homos[6] * cw + src_homos[7] * ch + 1.f;
      if(projection == 0.0f){
        continue;
      }
      const T src_w = (src_homos[0] * cw + src_homos[1] * ch + src_homos[2]) / projection;
      const T src_h = (src_homos[3] * cw + src_homos[4] * ch + src_homos[5]) / projection;

      if (src_h > 0.0f && src_w > 0.0f &&
        src_h < static_cast<T>(image_height - 1) && src_w < static_cast<T>(image_width - 1)) {
        const T fh = std::floor(src_h);
        const T fw = std::floor(src_w);
        const T dh = src_h - fh;
        const T dw = src_w - fw;
        const T coef_ff = dh*dw;
        const T coef_fc = dh*(1 - dw);
        const T coef_cc = (1 - dh)*(1 - dw);
        const T coef_cf = (1 - dh)*dw;

        const T *src_channels_ff = &images_data[((cn*image_height + static_cast<INDEX_TYPE>(fh))*image_width+static_cast<INDEX_TYPE>(fw))*image_channels];
        const T *src_channels_fc = &src_channels_ff[image_channels];
        const T *src_channels_cc = &src_channels_fc[img_height_step];
        const T *src_channels_cf = &src_channels_cc[image_channels];

        for(int cc = 0; cc < image_channels; cc++){
          T src_sample = coef_cc*src_channels_ff[cc] + coef_cf*src_channels_fc[cc] +
                                  coef_ff*src_channels_cc[cc] + coef_fc*src_channels_cf[cc];
          T diff = src_sample - ref_channels[cc];
          out_channels[cc] += diff*diff;
        }
        used_sample = used_sample + 1;
      }
    }
    out_mask_data[i] = used_sample;
    if(used_sample > 0.5f){
      for(int cc = 0; cc < image_channels; cc++){
        out_channels[cc] = out_channels[cc]/used_sample;
      }
    } else {
      for(int cc = 0; cc < image_channels; cc++){
        out_channels[cc] = 100;
      }
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
template <typename T, Interpolation INTERPOLATION_TYPE>
void CostVolumeFunctor<Eigen::GpuDevice, T, INTERPOLATION_TYPE>::operator()(
    const GPUDevice& d, const Tensor& images, const Tensor& transforms, Tensor* output, Tensor* output_mask) {
    const int64 batch_size = output->dim_size(0);
    const int64 image_height = output->dim_size(1);
    const int64 image_width = output->dim_size(2);
    const int64 image_depth = output->dim_size(3);
    const int64 image_channels = output->dim_size(4);
    const int64 image_num = images.dim_size(1);

    const int64 loop_count = batch_size * image_depth* image_height * image_width;
    const int64 input_image_size = batch_size * image_num * image_height * image_width * image_channels;
    const int64 output_cost_size = batch_size * image_height * image_width * image_depth * image_channels;

    if((input_image_size > INT32_MAX) || (output_cost_size > INT32_MAX)){
      if(batch_size == 1){
        auto config = GetGpuLaunchConfigBig(loop_count, d, CostVolumeKernelNoBatch<T, int64, INTERPOLATION_TYPE>, 0, 0);
        CostVolumeKernelNoBatch<T, int64, INTERPOLATION_TYPE><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                                  loop_count, batch_size, image_height, image_width, image_depth, image_channels, image_num,
                                  images.tensor<T, 5>().data(), transforms.tensor<T, 4>().data(), output->tensor<T, 5>().data(), output_mask->tensor<T, 5>().data());
      } else {
        auto config = GetGpuLaunchConfigBig(loop_count, d, CostVolumeKernel<T, int64, INTERPOLATION_TYPE>, 0, 0);
        CostVolumeKernel<T, int64, INTERPOLATION_TYPE><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                                  loop_count, batch_size, image_height, image_width, image_depth, image_channels, image_num,
                                  images.tensor<T, 5>().data(), transforms.tensor<T, 4>().data(), output->tensor<T, 5>().data(), output_mask->tensor<T, 5>().data());
      }
    } else {
      if(batch_size == 1){
        auto config = GetGpuLaunchConfigBig(loop_count, d, CostVolumeKernelNoBatch<T, int, INTERPOLATION_TYPE>, 0, 0);
        CostVolumeKernelNoBatch<T, int, INTERPOLATION_TYPE><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                                  loop_count, batch_size, image_height, image_width, image_depth, image_channels, image_num,
                                  images.tensor<T, 5>().data(), transforms.tensor<T, 4>().data(), output->tensor<T, 5>().data(), output_mask->tensor<T, 5>().data());
      } else {
        auto config = GetGpuLaunchConfigBig(loop_count, d, CostVolumeKernel<T, int, INTERPOLATION_TYPE>, 0, 0);
        CostVolumeKernel<T, int, INTERPOLATION_TYPE><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                                  loop_count, batch_size, image_height, image_width, image_depth, image_channels, image_num,
                                  images.tensor<T, 5>().data(), transforms.tensor<T, 4>().data(), output->tensor<T, 5>().data(), output_mask->tensor<T, 5>().data());
      }
    }
}

template struct CostVolumeFunctor<GPUDevice, float, INTERPOLATION_BILINEAR>;

template <typename T, typename INDEX_TYPE, Interpolation INTERPOLATION_TYPE>
__global__ void CostVolumeGradKernel(const INDEX_TYPE virtual_thread, 
              const INDEX_TYPE batch_size, const INDEX_TYPE image_height, const INDEX_TYPE image_width, 
              const INDEX_TYPE image_depth, const INDEX_TYPE image_channels, const INDEX_TYPE image_num,
              const T* images_data, const T* transforms_data, const T* transformed_mask_data, const T* grad_data, T* out_data){
  const INDEX_TYPE img_height_step = image_width*image_channels;
  const INDEX_TYPE img_step =  image_height*img_height_step;
  const INDEX_TYPE batch_img_step = image_num*img_step;
  const INDEX_TYPE homos_step = image_depth*8;
  const INDEX_TYPE batch_homos_step = (image_num - 1)*homos_step;

  for (auto i : GpuGridRangeX<INDEX_TYPE>(virtual_thread)){
    if(transformed_mask_data[i] < 0.5){
      continue;
    }
    auto tmp = i/image_depth;
    const auto cd = i - tmp*image_depth;
    auto tmp1 = tmp;
    tmp = tmp/image_width;
    const auto cw = tmp1 - tmp*image_width;
    const auto cb = tmp/image_height;
    const auto ch = tmp - cb*image_height;

    const T* batch_img_ptr = &images_data[cb*batch_img_step];
    T* batch_out_ptr = &out_data[cb*batch_img_step];
    const T *ref_channels = &batch_img_ptr[(ch*image_width+cw)*image_channels];
    T *ref_out_channels = &batch_out_ptr[(ch*image_width+cw)*image_channels];
    const T * curr_homos = &transforms_data[cb*batch_homos_step + cd*8];

    const T *grad_channels = &grad_data[i*image_channels];

    for(INDEX_TYPE cn = 1; cn < image_num; cn++){
      const T *src_homos = &curr_homos[(cn - 1)*homos_step];
      T projection = src_homos[6] * cw + src_homos[7] * ch + 1.f;
      if(projection == 0.0f){
        continue;
      }
      const T src_w = (src_homos[0] * cw + src_homos[1] * ch + src_homos[2]) / projection;
      const T src_h = (src_homos[3] * cw + src_homos[4] * ch + src_homos[5]) / projection;

      if (src_h > 0.0f && src_w > 0.0f &&
        src_h < static_cast<T>(image_height - 1) && src_w < static_cast<T>(image_width - 1)) {
        const T fh = std::floor(src_h);
        const T fw = std::floor(src_w);
        const T dh = src_h - fh;
        const T dw = src_w - fw;
        const T coef_ff = dh*dw;
        const T coef_fc = dh*(1 - dw);
        const T coef_cc = (1 - dh)*(1 - dw);
        const T coef_cf = (1 - dh)*dw;

        const T *src_channels_ff = &batch_img_ptr[((cn*image_height + static_cast<INDEX_TYPE>(fh))*image_width+static_cast<INDEX_TYPE>(fw))*image_channels];
        T *src_out_channels_ff = &batch_out_ptr[((cn*image_height + static_cast<INDEX_TYPE>(fh))*image_width+static_cast<INDEX_TYPE>(fw))*image_channels];
        const T *src_channels_fc = &src_channels_ff[image_channels];
        T *src_out_channels_fc = &src_out_channels_ff[image_channels];
        const T *src_channels_cc = &src_channels_fc[img_height_step];
        T *src_out_channels_cc = &src_out_channels_fc[img_height_step];
        const T *src_channels_cf = &src_channels_cc[image_channels];
        T *src_out_channels_cf = &src_out_channels_cc[image_channels];

        for(int cc = 0; cc < image_channels; cc++){
          T src_sample = coef_cc*src_channels_ff[cc] + coef_cf*src_channels_fc[cc] +
                                  coef_ff*src_channels_cc[cc] + coef_fc*src_channels_cf[cc];
          T diff = src_sample - ref_channels[cc];
          T ref_grad = 2*diff*grad_channels[cc]/transformed_mask_data[i];
          atomicAdd(&ref_out_channels[cc], -ref_grad);
          atomicAdd(&src_out_channels_ff[cc], ref_grad*coef_cc);
          atomicAdd(&src_out_channels_fc[cc], ref_grad*coef_cf);
          atomicAdd(&src_out_channels_cc[cc], ref_grad*coef_ff);
          atomicAdd(&src_out_channels_cf[cc], ref_grad*coef_fc);
        }
      }
    }
  }
}

template <typename T, typename INDEX_TYPE, Interpolation INTERPOLATION_TYPE>
__global__ void CostVolumeGradKernelNoBatch(const INDEX_TYPE virtual_thread, 
              const INDEX_TYPE batch_size, const INDEX_TYPE image_height, const INDEX_TYPE image_width, 
              const INDEX_TYPE image_depth, const INDEX_TYPE image_channels, const INDEX_TYPE image_num,
              const T* images_data, const T* transforms_data, const T* transformed_mask_data, const T* grad_data, T* out_data){
  (void )batch_size;
  const INDEX_TYPE img_height_step = image_width*image_channels;
  const INDEX_TYPE homos_step = image_depth*8;

  for (auto i : GpuGridRangeX<INDEX_TYPE>(virtual_thread)){
    if(transformed_mask_data[i] < 0.5){
      continue;
    }
    auto tmp = i/image_depth;
    const auto cd = i - tmp*image_depth;
    const auto ch = tmp/image_width;
    const auto cw = tmp - ch*image_width;

    const T *ref_channels = &images_data[(ch*image_width+cw)*image_channels];
    T *ref_out_channels = &out_data[(ch*image_width+cw)*image_channels];
    const T *  curr_homos = &transforms_data[cd*8];

    const T *grad_channels = &grad_data[i*image_channels];

    for(INDEX_TYPE cn = 1; cn < image_num; cn++){
      const T *src_homos = &curr_homos[(cn - 1)*homos_step];
      T projection = src_homos[6] * cw + src_homos[7] * ch + 1.f;
      if(projection == 0.0f){
        continue;
      }
      const T src_w = (src_homos[0] * cw + src_homos[1] * ch + src_homos[2]) / projection;
      const T src_h = (src_homos[3] * cw + src_homos[4] * ch + src_homos[5]) / projection;

      if (src_h > 0.0f && src_w > 0.0f &&
        src_h < static_cast<T>(image_height - 1) && src_w < static_cast<T>(image_width - 1)) {
        const T fh = std::floor(src_h);
        const T fw = std::floor(src_w);
        const T dh = src_h - fh;
        const T dw = src_w - fw;
        const T coef_ff = dh*dw;
        const T coef_fc = dh*(1 - dw);
        const T coef_cc = (1 - dh)*(1 - dw);
        const T coef_cf = (1 - dh)*dw;

        const T *src_channels_ff = &images_data[((cn*image_height + static_cast<INDEX_TYPE>(fh))*image_width+static_cast<INDEX_TYPE>(fw))*image_channels];
        T *src_out_channels_ff = &out_data[((cn*image_height + static_cast<INDEX_TYPE>(fh))*image_width+static_cast<INDEX_TYPE>(fw))*image_channels];
        const T *src_channels_fc = &src_channels_ff[image_channels];
        T *src_out_channels_fc = &src_out_channels_ff[image_channels];
        const T *src_channels_cc = &src_channels_fc[img_height_step];
        T *src_out_channels_cc = &src_out_channels_fc[img_height_step];
        const T *src_channels_cf = &src_channels_cc[image_channels];
        T *src_out_channels_cf = &src_out_channels_cc[image_channels];

        for(int cc = 0; cc < image_channels; cc++){
          T src_sample = coef_cc*src_channels_ff[cc] + coef_cf*src_channels_fc[cc] +
                                  coef_ff*src_channels_cc[cc] + coef_fc*src_channels_cf[cc];
          T diff = src_sample - ref_channels[cc];
          T ref_grad = 2*diff*grad_channels[cc]/transformed_mask_data[i];
          atomicAdd(&ref_out_channels[cc], -ref_grad);
          atomicAdd(&src_out_channels_ff[cc], ref_grad*coef_cc);
          atomicAdd(&src_out_channels_fc[cc], ref_grad*coef_cf);
          atomicAdd(&src_out_channels_cc[cc], ref_grad*coef_ff);
          atomicAdd(&src_out_channels_cf[cc], ref_grad*coef_fc);
        }
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
// Define the GPU implementation that launches the CUDA kernel.
template <typename T, Interpolation INTERPOLATION_TYPE>
void CostVolumeGradFunctor<Eigen::GpuDevice, T, INTERPOLATION_TYPE>::operator()(
    const GPUDevice& d, const Tensor& images, const Tensor& transforms, const Tensor& transformed_mask, const Tensor& grad, Tensor* output) {
    const int64 batch_size = grad.dim_size(0);
    const int64 image_height = grad.dim_size(1);
    const int64 image_width = grad.dim_size(2);
    const int64 image_depth = grad.dim_size(3);
    const int64 image_channels = grad.dim_size(4);
    const int64 image_num = images.dim_size(1);

    const int64 loop_count = batch_size * image_depth* image_height * image_width;
    const int64 input_image_size = batch_size * image_num * image_height * image_width * image_channels;
    const int64 output_cost_size = batch_size * image_height * image_width * image_depth * image_channels;

    if((input_image_size > INT32_MAX) || (output_cost_size > INT32_MAX)){
      auto config = GetGpuLaunchConfigBig(input_image_size, d, SetZeroBig<T, int64>, 0, 0);
      SetZeroBig<T, int64><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(input_image_size, output->tensor<T, 5>().data());
      if(batch_size == 1){
        config = GetGpuLaunchConfigBig(loop_count, d, CostVolumeGradKernelNoBatch<T, int64, INTERPOLATION_TYPE>, 0, 0);
        CostVolumeGradKernelNoBatch<T, int64, INTERPOLATION_TYPE><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                                  loop_count, batch_size, image_height, image_width, image_depth, image_channels, image_num,
                                  images.tensor<T, 5>().data(), transforms.tensor<T, 4>().data(), transformed_mask.tensor<T, 5>().data(), 
                                  grad.tensor<T, 5>().data(), output->tensor<T, 5>().data());
      } else {
        config = GetGpuLaunchConfigBig(loop_count, d, CostVolumeGradKernel<T, int64, INTERPOLATION_TYPE>, 0, 0);
        CostVolumeGradKernel<T, int64, INTERPOLATION_TYPE><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                                  loop_count, batch_size, image_height, image_width, image_depth, image_channels, image_num,
                                  images.tensor<T, 5>().data(), transforms.tensor<T, 4>().data(), transformed_mask.tensor<T, 5>().data(),
                                  grad.tensor<T, 5>().data(), output->tensor<T, 5>().data());
      }
    } else {
      auto config = GetGpuLaunchConfigBig(input_image_size, d, SetZeroBig<T, int>, 0, 0);
      SetZeroBig<T, int><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(input_image_size, output->tensor<T, 5>().data());
      if(batch_size == 1){
        config = GetGpuLaunchConfigBig(loop_count, d, CostVolumeGradKernelNoBatch<T, int, INTERPOLATION_TYPE>, 0, 0);
        CostVolumeGradKernelNoBatch<T, int, INTERPOLATION_TYPE><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                                  loop_count, batch_size, image_height, image_width, image_depth, image_channels, image_num,
                                  images.tensor<T, 5>().data(), transforms.tensor<T, 4>().data(), transformed_mask.tensor<T, 5>().data(),
                                  grad.tensor<T, 5>().data(), output->tensor<T, 5>().data());
      } else {
        config = GetGpuLaunchConfigBig(loop_count, d, CostVolumeGradKernel<T, int, INTERPOLATION_TYPE>, 0, 0);
        CostVolumeGradKernel<T, int, INTERPOLATION_TYPE><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                                  loop_count, batch_size, image_height, image_width, image_depth, image_channels, image_num,
                                  images.tensor<T, 5>().data(), transforms.tensor<T, 4>().data(), transformed_mask.tensor<T, 5>().data(),
                                  grad.tensor<T, 5>().data(), output->tensor<T, 5>().data());
      }
    }
}

template struct CostVolumeGradFunctor<GPUDevice, float, INTERPOLATION_BILINEAR>;

}  // end namespace functor

}  // end namespace addons
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
