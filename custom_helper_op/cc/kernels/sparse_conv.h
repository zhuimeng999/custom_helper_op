#ifndef _INDEX_INITIALIZER_H
#define _INDEX_INITIALIZER_H

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace custom_helper_op {
namespace functor {

template <typename Device, typename T>
struct SparseConv2DFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, 
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
              T * out_data);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct SparseConv2DFunctor<Eigen::GpuDevice, T> {
void operator()(const Eigen::GpuDevice& d, 
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
              T * out_data);
};
#endif

template <typename Device, typename T>
struct SparseConv2DGradFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, 
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
              T * default_channel_value_grad);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct SparseConv2DGradFunctor<Eigen::GpuDevice, T> {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Eigen::GpuDevice& d, 
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
              T * default_channel_value_grad);
};
#endif

} /* functor */
} /* custom_helper_op */
} /* tensorflow */
#endif /* _INDEX_INITIALIZER_H */