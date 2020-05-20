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
#include "tensorflow_cost_volume/cc/kernels/cost_volume.h"

namespace tensorflow {
namespace addons {

namespace functor {

// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;

template class FillProjectiveTransform<GPUDevice, uint8>;
template class FillProjectiveTransform<GPUDevice, int32>;
template class FillProjectiveTransform<GPUDevice, int64>;
template class FillProjectiveTransform<GPUDevice, Eigen::half>;
template class FillProjectiveTransform<GPUDevice, float>;
template class FillProjectiveTransform<GPUDevice, double>;

// Define the CUDA kernel.
template <typename T>
__global__ void CostVolumeKernel(int virtual_thread, 
            const int batch_size, const int image_height, const int image_width, const int image_depth, const int image_channels, 
            const int image_num, const T* images_data, const float* transforms_data, T* out) { 
  int img_step =  image_height*image_width*image_channels;
  int batch_img_step = image_num*img_step;
  int homos_step = image_depth*8;
  int batch_homos_step = (image_num - 1)*homos_step;
  for (auto i : GpuGridRangeX<int>(virtual_thread)){
    int tmp = i/image_depth;
    int cd = i - tmp*image_depth;
    int tmp1 = tmp;
    tmp = tmp/image_width;
    int cw = tmp1 - tmp*image_width;
    int cb = tmp/image_height;
    int ch = tmp - cb*image_height;

    const T *in_ptr = &images_data[cb*batch_img_step];
    const T *ref_channels = &in_ptr[(ch*image_width+cw)*image_channels];
    const float *  curr_homos = &transforms_data[cb*batch_homos_step + cd*8];
    T *out_channels = &out[i*image_channels];
    for(int cc = 0; cc < image_channels; cc++){
      out_channels[cc] = T(0);
    }
    for(int cn = 1; cn < image_num; cn++){
      const float *src_homos = &curr_homos[(cn - 1)*homos_step];

      float projection = src_homos[6] * cw + src_homos[7] * ch + 1.f;
      if(projection == 0.0f){
        continue;
      }
      const float src_w = (src_homos[0] * cw + src_homos[1] * ch + src_homos[2]) / projection;
      const float src_h = (src_homos[3] * cw + src_homos[4] * ch + src_homos[5]) / projection;

      const int w_near = __float2int_rn(src_w);
      const int h_near = __float2int_rn(src_h);
      if((h_near < 0) || (h_near >=image_height) || (w_near < 0) || (w_near >= image_height)){
        continue;
      }
      const T *src_channels = &in_ptr[((cn*image_height + h_near)*image_width+w_near)*image_channels];
      for(int cc = 0; cc < image_channels; cc++){
        T diff = src_channels[cc] - ref_channels[cc];
        out_channels[cc] += diff*diff;
      }
    }
  }
}


// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void CostVolumeFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, const Tensor& images, const Tensor& transforms, Tensor* output) {
    const int batch_size = output->dim_size(0);
    const int image_height = output->dim_size(1);
    const int image_width = output->dim_size(2);
    const int image_depth = output->dim_size(3);
    const int image_channels = output->dim_size(4);

    const auto output_data_size =
        static_cast<int64>(batch_size) * image_height * image_width * image_depth;
    CHECK_LE(output_data_size, 0xFFFFFFFFUL);
    GpuLaunchConfig config = GetGpuLaunchConfig(output_data_size, d);
    TF_CHECK_OK(GpuLaunchKernel(
        CostVolumeKernel<T>, config.block_count, config.thread_per_block, 0, d.stream(), output_data_size, 
        batch_size, image_height, image_width, image_depth, image_channels, images.dim_size(1),
        images.tensor<T, 5>().data(), transforms.tensor<float, 4>().data(), output->tensor<T, 5>().data()));
}

template struct CostVolumeFunctor<GPUDevice, uint8>;
template struct CostVolumeFunctor<GPUDevice, int32>;
template struct CostVolumeFunctor<GPUDevice, int64>;
//template struct CostVolumeFunctor<GPUDevice, Eigen::half>;
template struct CostVolumeFunctor<GPUDevice, float>;
template struct CostVolumeFunctor<GPUDevice, double>;

}  // end namespace functor

}  // end namespace addons
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
