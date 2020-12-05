#ifndef _INDEX_INITIALIZER_H
#define _INDEX_INITIALIZER_H

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace custom_helper_op {
namespace functor {

#define SPARSE_CONV_BASE_ARG_DEF_LIST \
              const int stride_h, \
              const int stride_w, \
              const int dilation_h, \
              const int dilations_w, \
              const int filter_h, \
              const int filter_w, \
              const int64 batch_size, \
              const int64 image_height, \
              const int64 image_width, \
              const int64 image_channels, \
              const int64 out_channels, \
              const T* images_data, \
              const T* filter_data, \
              const T* base_plane_data,\
              const T* default_channel_value,\
              const T* offsets_data

template <typename Device, typename T>
struct SparseConv2DFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, 
              SPARSE_CONV_BASE_ARG_DEF_LIST,
              T * out_data);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct SparseConv2DFunctor<Eigen::GpuDevice, T> {
void operator()(const Eigen::GpuDevice& d, 
              SPARSE_CONV_BASE_ARG_DEF_LIST,
              T * out_data);
};
#endif

template <typename Device, typename T>
struct SparseConv2DGradFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, 
              SPARSE_CONV_BASE_ARG_DEF_LIST,
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
              SPARSE_CONV_BASE_ARG_DEF_LIST,
              const T * out_grad_data,
              T * images_grad_data,
              T * filter_grad_data,
              T * base_plane_grad_data,
              T * default_channel_value_grad);
};
#endif

#define SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST \
        int kKnownFilterHeight, int kKnownFilterWidth, int kKnownFilterDepth, \
        int kKnownDilationHeight, int kKnownDilationWidth, int kKnownDilationDepth, \
        int kKnownStrideHeight, int kKnownStrideWidth, int kKnownStrideDepth

#define SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST \
        kKnownFilterHeight, kKnownFilterWidth, kKnownFilterDepth, \
        kKnownDilationHeight, kKnownDilationWidth, kKnownDilationDepth, \
        kKnownStrideHeight, kKnownStrideWidth, kKnownStrideDepth

#define SPARSE_CONV3D_BASE_ARG_DEF_LIST \
              const int stride_h, \
              const int stride_w, \
              const int stride_d, \
              const int dilations_h, \
              const int dilations_w, \
              const int dilations_d, \
              const int filter_h, \
              const int filter_w, \
              const int filter_d, \
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


template <typename Device, typename T, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
struct SparseConv3DFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, 
              SPARSE_CONV3D_BASE_ARG_DEF_LIST,
              T * out_data);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
struct SparseConv3DFunctor<Eigen::GpuDevice, T, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST> {
void operator()(const Eigen::GpuDevice& d, 
              SPARSE_CONV3D_BASE_ARG_DEF_LIST,
              T * out_data);
};
#endif

template <typename Device, typename T, bool dynamic_default, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
struct SparseConv3DGradFunctor {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Device& d, 
              SPARSE_CONV3D_BASE_ARG_DEF_LIST,
              const T * out_grad_data,
              T * images_grad_data,
              T * filter_grad_data,
              T * default_channel_value_grad);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T, bool dynamic_default, SPARSE_CONV3D_FIX_PARAMETOR_DEF_LIST>
struct SparseConv3DGradFunctor<Eigen::GpuDevice, T, dynamic_default, SPARSE_CONV3D_FIX_PARAMETOR_ARG_LIST> {
// Computes on device "d": out = out.constant(in(0)),
void operator()(const Eigen::GpuDevice& d,
              SPARSE_CONV3D_BASE_ARG_DEF_LIST,
              const T * out_grad_data,
              T * images_grad_data,
              T * filter_grad_data,
              T * default_channel_value_grad);
};
#endif

} /* functor */
} /* custom_helper_op */
} /* tensorflow */
#endif /* _INDEX_INITIALIZER_H */