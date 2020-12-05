#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "custom_helper_op/cc/kernels/sparse_conv.h"

namespace tensorflow {
namespace custom_helper_op {

using functor::SparseConv2DFunctor;

template <typename Device, typename T>
class SparseConv2DOp : public OpKernel {
 private:
  std::vector<int> strides_;
  std::vector<int> dilations_;
 public:
  explicit SparseConv2DOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES(ctx, strides_.size() == 2, errors::InvalidArgument("strides must be a vector have 2 element"));
    OP_REQUIRES(ctx, dilations_.size() == 2, errors::InvalidArgument("dilations must be a vector have 2 element"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& images = ctx->input(0);
    const Tensor& filter = ctx->input(1);
    const Tensor& base_plane = ctx->input(2);
    const Tensor& default_channels_value = ctx->input(3);
    const Tensor& offsets = ctx->input(4);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(default_channels_value.shape()),
                errors::InvalidArgument("default_value must be scalar: ",
                                        default_channels_value.shape().DebugString()));
    OP_REQUIRES(ctx, images.shape().dims() == 4,
                errors::InvalidArgument("images must have rank 4"));

    const auto batch_size = images.dim_size(0);
    const auto image_height = images.dim_size(1);
    const auto image_width = images.dim_size(2);
    const auto image_channels = images.dim_size(3);

    OP_REQUIRES(ctx, (filter.shape().dims() == 4) && (filter.dim_size(3) == image_channels),
                errors::InvalidArgument("filter must have rank 4, and must compate to ref_image"));
    const auto out_channels = filter.dim_size(0);
    const auto filter_h = filter.dim_size(1);
    const auto filter_w = filter.dim_size(2);

    OP_REQUIRES(ctx, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == batch_size)
                      && (base_plane.dim_size(1) == image_height) && (base_plane.dim_size(2) == image_width) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to ref_image"));
    OP_REQUIRES(ctx, (offsets.shape().dims() == 2) && (offsets.dim_size(0) == batch_size),
                errors::InvalidArgument("offsets must have rank 2, and must compate to ref_image"));

    Tensor *output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            TensorShape({batch_size, image_height, image_width, out_channels}),
                            &output));


    SparseConv2DFunctor<Device, T>()(ctx->eigen_device<Device>(), 
                  strides_[0], 
                  strides_[1],
                  dilations_[0],
                  dilations_[1],
                  filter_h,
                  filter_w,
                  batch_size,
                  image_height,
                  image_width,
                  image_channels,
                  out_channels,
                  images.tensor<T, 4>().data(),
                  filter.tensor<T, 4>().data(), 
                  base_plane.tensor<T, 4>().data(),
                  default_channels_value.flat<T>().data(),
                  offsets.tensor<T, 2>().data(),
                  output->tensor<T, 4>().data());

  }
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("SparseConv2D") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          SparseConv2DOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

using functor::SparseConv2DGradFunctor;

template <typename Device, typename T>
class SparseConv2DGradOp : public OpKernel {
 private:
  std::vector<int> strides_;
  std::vector<int> dilations_;
 public:
  explicit SparseConv2DGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES(ctx, strides_.size() == 2, errors::InvalidArgument("strides must be a vector have 2 element"));
    OP_REQUIRES(ctx, dilations_.size() == 2, errors::InvalidArgument("dilations must be a vector have 2 element"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& images = ctx->input(0);
    const Tensor& filter = ctx->input(1);
    const Tensor& base_plane = ctx->input(2);
    const Tensor& default_channels_value = ctx->input(3);
    const Tensor& offsets = ctx->input(4);
    const Tensor& out_grad = ctx->input(5);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(default_channels_value.shape()),
                errors::InvalidArgument("default_value must be scalar: ",
                                        default_channels_value.shape().DebugString()));
    OP_REQUIRES(ctx, images.shape().dims() == 4,
                errors::InvalidArgument("images must have rank 4"));

    const auto batch_size = images.dim_size(0);
    const auto image_height = images.dim_size(1);
    const auto image_width = images.dim_size(2);
    const auto image_channels = images.dim_size(3);

    OP_REQUIRES(ctx, (filter.shape().dims() == 4) && (filter.dim_size(3) == image_channels),
                errors::InvalidArgument("filter must have rank 4, and must compate to ref_image"));
    const auto out_channels = filter.dim_size(0);
    const auto filter_h = filter.dim_size(1);
    const auto filter_w = filter.dim_size(2);

    OP_REQUIRES(ctx, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == batch_size)
                      && (base_plane.dim_size(1) == image_height) && (base_plane.dim_size(2) == image_width) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to ref_image"));
    OP_REQUIRES(ctx, (offsets.shape().dims() == 2) && (offsets.dim_size(0) == batch_size),
                errors::InvalidArgument("offsets must have rank 2, and must compate to ref_image"));

    OP_REQUIRES(ctx, (out_grad.shape().dims() == 4) && (out_grad.dim_size(0) == batch_size)
                      && (out_grad.dim_size(1) == image_height) && (out_grad.dim_size(2) == image_width) && (out_grad.dim_size(3) == out_channels),
                errors::InvalidArgument("out_grad must have rank 4, and must compate to ref_image, got ", out_grad.shape().DebugString()));

    Tensor *images_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            images.shape(),
                            &images_grad));
    Tensor *filter_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1,
                            filter.shape(),
                            &filter_grad));
    Tensor *baze_plane_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            2,
                            base_plane.shape(),
                            &baze_plane_grad));
    Tensor *default_channels_value_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            3,
                            default_channels_value.shape(),
                            &default_channels_value_grad));


    SparseConv2DGradFunctor<Device, T>()(ctx->eigen_device<Device>(), 
                  strides_[0], 
                  strides_[1],
                  dilations_[0],
                  dilations_[1],
                  filter_h,
                  filter_w,
                  batch_size,
                  image_height,
                  image_width,
                  image_channels,
                  out_channels,
                  images.tensor<T, 4>().data(),
                  filter.tensor<T, 4>().data(), 
                  base_plane.tensor<T, 4>().data(),
                  default_channels_value.flat<T>().data(),
                  offsets.tensor<T, 2>().data(),
                  out_grad.tensor<T, 4>().data(),
                  images_grad->tensor<T, 4>().data(),
                  filter_grad->tensor<T, 4>().data(),
                  baze_plane_grad->tensor<T, 4>().data(),
                  default_channels_value_grad->flat<T>().data());

  }
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("SparseConv2DGrad") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          SparseConv2DGradOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
// TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

using functor::SparseConv3DFunctor;

#define SPARSE_CONV3D_FUNCTOR_BASE_ARG_LIST \
                    strides_[0],\
                    strides_[1],\
                    strides_[2],\
                    dilations_[0],\
                    dilations_[1],\
                    dilations_[2],\
                    filter_h,\
                    filter_w,\
                    filter_d,\
                    batch_size,\
                    image_height,\
                    image_width,\
                    image_depth,\
                    image_channels,\
                    out_height,\
                    out_width,\
                    out_depth,\
                    out_channel_num,\
                    images.tensor<T, 5>().data(),\
                    filter.tensor<T, 5>().data(), \
                    default_channels_value.flat<T>().data(),\
                    base_plane.tensor<int32, 4>().data()

#define SPARSE_CONV3D_KERNEL_CALL() \
    if((dilations_[0] == 1) && (dilations_[1] == 1) && (dilations_[2] == 1)){ \
      if((filter_h == 3) && (filter_w == 3) && (filter_h == 3)){ \
        FIXED_FILTER_SIZE_CASE(3, 3, 3, 1, 1, 1); \
      } else { \
        FIXED_FILTER_SIZE_CASE(-1, -1, -1, 1, 1, 1);\
      }\
    } else if((dilations_[0] == 2) && (dilations_[1] == 2) && (dilations_[2] == 2)){\
      if((filter_h == 3) && (filter_w == 3) && (filter_h == 3)){\
        FIXED_FILTER_SIZE_CASE(3, 3, 3, 2, 2, 2);\
      } else {\
        FIXED_FILTER_SIZE_CASE(-1, -1, -1, 2, 2, 2);\
      }\
    } else {\
      if((filter_h == 3) && (filter_w == 3) && (filter_h == 3)){\
        FIXED_FILTER_SIZE_CASE(3, 3, 3, -1, -1, -1);\
      } else {\
        FIXED_FILTER_SIZE_CASE(-1, -1, -1, -1, -1, -1);\
      }\
    }


template <typename Device, typename T>
class SparseConv3DOp : public OpKernel {
 private:
  std::vector<int> strides_;
  std::vector<int> dilations_;
  bool dynamic_default_;
 public:
  explicit SparseConv3DOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_default", &dynamic_default_));
    OP_REQUIRES(ctx, strides_.size() == 3, errors::InvalidArgument("strides must be a vector have 3 element"));
    OP_REQUIRES(ctx, dilations_.size() == 3, errors::InvalidArgument("dilations must be a vector have 3 element"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& images = ctx->input(0);
    const Tensor& filter = ctx->input(1);
    const Tensor& default_channels_value = ctx->input(2);
    const Tensor& base_plane = ctx->input(3);


    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(default_channels_value.shape()),
                errors::InvalidArgument("default_value must be scalar: ",
                                        default_channels_value.shape().DebugString()));
    OP_REQUIRES(ctx, images.shape().dims() == 5,
                errors::InvalidArgument("images must have rank 4"));

    const auto batch_size = images.dim_size(0);
    const auto image_height = images.dim_size(1);
    const auto image_width = images.dim_size(2);
    const auto image_depth = images.dim_size(3);
    const auto image_channels = images.dim_size(4);

    OP_REQUIRES(ctx, (filter.shape().dims() == 5) && (filter.dim_size(3) == image_channels),
                errors::InvalidArgument("filter must have rank 5, and must compate to ref_image"));
    const auto filter_h = filter.dim_size(0);
    const auto filter_w = filter.dim_size(1);
    const auto filter_d = filter.dim_size(2);
    const auto out_channel_num = filter.dim_size(4);

    OP_REQUIRES(ctx, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == batch_size)
                      && (base_plane.dim_size(1) == image_height) && (base_plane.dim_size(2) == image_width) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to ref_image, got ", base_plane.shape().DebugString()));

    const auto out_height = (image_height + strides_[0] - 1)/strides_[0];
    const auto out_width = (image_width + strides_[1] - 1)/strides_[1];
    const auto out_depth = (image_depth + strides_[2] - 1)/strides_[2];
    Tensor *output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            TensorShape({batch_size, out_height, out_width, out_depth, out_channel_num}),
                            &output));


    #define FIXED_FILTER_SIZE_CASE(f_h, f_w, f_d, d_h, d_w, d_d) \
      SparseConv3DFunctor<Device, T, f_h, f_w, f_d, d_h, d_w, d_d, -1, -1, -1>()(ctx->eigen_device<Device>(), \
                    SPARSE_CONV3D_FUNCTOR_BASE_ARG_LIST, \
                    output->tensor<T, 5>().data());
    SPARSE_CONV3D_KERNEL_CALL();
    #undef FIXED_FILTER_SIZE_CASE
  }
private:
  TF_DISALLOW_COPY_AND_ASSIGN(SparseConv3DOp);
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("SparseConv3D") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          SparseConv3DOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
// TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

using functor::SparseConv3DGradFunctor;

template <typename Device, typename T>
class SparseConv3DGradOp : public OpKernel {
 private:
  std::vector<int> strides_;
  std::vector<int> dilations_;
  bool dynamic_default_;
 public:
  explicit SparseConv3DGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_default", &dynamic_default_));
    OP_REQUIRES(ctx, strides_.size() == 3, errors::InvalidArgument("strides must be a vector have 3 element"));
    OP_REQUIRES(ctx, dilations_.size() == 3, errors::InvalidArgument("dilations must be a vector have 3 element"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& images = ctx->input(0);
    const Tensor& filter = ctx->input(1);
    const Tensor& default_channels_value = ctx->input(2);
    const Tensor& base_plane = ctx->input(3);

    const Tensor& out_grad = ctx->input(4);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(default_channels_value.shape()),
                errors::InvalidArgument("default_value must be scalar: ",
                                        default_channels_value.shape().DebugString()));
    OP_REQUIRES(ctx, images.shape().dims() == 5,
                errors::InvalidArgument("images must have rank 5"));

    const auto batch_size = images.dim_size(0);
    const auto image_height = images.dim_size(1);
    const auto image_width = images.dim_size(2);
    const auto image_depth = images.dim_size(3);
    const auto image_channels = images.dim_size(4);

    OP_REQUIRES(ctx, (filter.shape().dims() == 5) && (filter.dim_size(3) == image_channels),
                errors::InvalidArgument("filter must have rank 5, and must compate to ref_image"));
    const auto filter_h = filter.dim_size(0);
    const auto filter_w = filter.dim_size(1);
    const auto filter_d = filter.dim_size(2);
    const auto out_channel_num = filter.dim_size(4);

    OP_REQUIRES(ctx, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == batch_size)
                      && (base_plane.dim_size(1) == image_height) && (base_plane.dim_size(2) == image_width) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to ref_image"));

    const auto out_height = (image_height + strides_[0] - 1)/strides_[0];
    const auto out_width = (image_width + strides_[1] - 1)/strides_[1];
    const auto out_depth = (image_depth + strides_[2] - 1)/strides_[2];
    OP_REQUIRES(ctx, (out_grad.shape().dims() == 5) && (out_grad.dim_size(0) == batch_size)
                      && (out_grad.dim_size(1) == out_height) && (out_grad.dim_size(2) == out_width) && (out_grad.dim_size(3) == out_depth)
                      && (out_grad.dim_size(4) == out_channel_num),
                errors::InvalidArgument("out_grad must have rank 5, and must compate to ref_image, got ", out_grad.shape().DebugString()));

    Tensor *images_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            images.shape(),
                            &images_grad));
    Tensor *filter_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1,
                            filter.shape(),
                            &filter_grad));
    Tensor *default_channels_value_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                              2,
                              default_channels_value.shape(),
                              &default_channels_value_grad));

    #define FIXED_FILTER_SIZE_CASE(f_h, f_w, f_d, d_h, d_w, d_d) \
      if(dynamic_default_){\
        SparseConv3DGradFunctor<Device, T, true, f_h, f_w, f_d, d_h, d_w, d_d, -1, -1, -1>()(ctx->eigen_device<Device>(), \
                      SPARSE_CONV3D_FUNCTOR_BASE_ARG_LIST,\
                      out_grad.tensor<T, 5>().data(),\
                      images_grad->tensor<T, 5>().data(),\
                      filter_grad->tensor<T, 5>().data(),\
                      default_channels_value_grad->flat<T>().data());\
      } else {\
        SparseConv3DGradFunctor<Device, T, false, f_h, f_w, f_d, d_h, d_w, d_d, -1, -1, -1>()(ctx->eigen_device<Device>(), \
                      SPARSE_CONV3D_FUNCTOR_BASE_ARG_LIST,\
                      out_grad.tensor<T, 5>().data(),\
                      images_grad->tensor<T, 5>().data(),\
                      filter_grad->tensor<T, 5>().data(),\
                      default_channels_value_grad->flat<T>().data());\
      }

    SPARSE_CONV3D_KERNEL_CALL();
    #undef FIXED_FILTER_SIZE_CASE
  }
private:
  TF_DISALLOW_COPY_AND_ASSIGN(SparseConv3DGradOp);
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("SparseConv3DGrad") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          SparseConv3DGradOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
// TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

} /* custom_helper_op */
} /* tensorflow */