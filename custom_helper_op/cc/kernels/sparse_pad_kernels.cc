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
#include "custom_helper_op/cc/kernels/sparse_pad.h"

namespace tensorflow {
namespace custom_helper_op {

namespace functor {

template <typename T>
struct SparsePadFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(OpKernelContext* ctx, const Eigen::ThreadPoolDevice& d, SPARSE_PAD_BASE_ARG_DEF_LIST, T* out_data)
{
    auto initializer_function = [&](const int start, const int limit) {
      for (int i = start; i < limit; ++i) {
        T *tmp = &out_data[i*3];
        tmp[0] = T(i%out_width);
        tmp[1] = T(i/out_width);
        tmp[2] = T(1.0);
      }
    };

    auto total_elm = out_height*out_width;

    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    auto schedulingParams = thread::ThreadPool::SchedulingParams(
                                thread::ThreadPool::SchedulingStrategy::kFixedBlockSize, 0, 
                                (total_elm + thread_pool->NumThreads() - 1)/thread_pool->NumThreads());
    thread_pool->ParallelFor(total_elm, schedulingParams, initializer_function);
}
};
template struct SparsePadFunctor<Eigen::ThreadPoolDevice, float>;
template struct SparsePadFunctor<Eigen::ThreadPoolDevice, double>;

template <typename T>
struct SparsePadGradFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(OpKernelContext* ctx, const Eigen::ThreadPoolDevice& d, SPARSE_PAD_BASE_ARG_DEF_LIST, const T* out_data, T* image_grad_data)
{
    auto initializer_function = [&](const int start, const int limit) {
      for (int i = start; i < limit; ++i) {
        T *tmp = &image_grad_data[i*3];
        tmp[0] = T(i%out_width);
        tmp[1] = T(i/out_width);
        tmp[2] = T(1.0);
      }
    };

    auto total_elm = out_height*out_width;

    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    auto schedulingParams = thread::ThreadPool::SchedulingParams(
                                thread::ThreadPool::SchedulingStrategy::kFixedBlockSize, 0, 
                                (total_elm + thread_pool->NumThreads() - 1)/thread_pool->NumThreads());
    thread_pool->ParallelFor(total_elm, schedulingParams, initializer_function);
}
};
template struct SparsePadGradFunctor<Eigen::ThreadPoolDevice, float>;
template struct SparsePadGradFunctor<Eigen::ThreadPoolDevice, double>;

} /* functor */

using functor::SparsePadFunctor;
using functor::SparsePadGradFunctor;

template <typename Device, typename T>
class SparsePadOp : public OpKernel {
 private:
  std::vector<int> strides_;
  std::vector<int> dilations_;
 public:
  explicit SparsePadOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES(ctx, strides_.size() == 3, errors::InvalidArgument("strides must be a vector have 3 element"));
    OP_REQUIRES(ctx, dilations_.size() == 3, errors::InvalidArgument("dilations must be a vector have 3 element"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& images = ctx->input(0);
    const Tensor& base_plane = ctx->input(1);

    const auto batch_size = images.dim_size(0);
    const auto image_height = images.dim_size(1);
    const auto image_width = images.dim_size(2);
    const auto image_depth = images.dim_size(3);
    const auto image_channels = images.dim_size(4);

    const auto out_height = image_height * strides_[0];
    const auto out_width = image_width * strides_[1];
    const auto out_depth = image_depth * strides_[2];

    OP_REQUIRES(ctx, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == batch_size)
                      && (base_plane.dim_size(1) == out_height) && (base_plane.dim_size(2) == out_width) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to ref_image, got ", base_plane.shape().DebugString()));

    Tensor *output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            TensorShape({batch_size, out_height, out_width, out_depth, image_channels}),
                            &output));


    SparsePadFunctor<Device, T>()(ctx, ctx->eigen_device<Device>(),
                                      strides_[0],
                                      strides_[1],
                                      strides_[2],
                                      batch_size,
                                      image_height,
                                      image_width,
                                      image_depth,
                                      image_channels,
                                      out_height,
                                      out_width,
                                      out_depth,
                                      images.tensor<T, 5>().data(),
                                      base_plane.tensor<int32, 4>().data(),
                                      output->tensor<T, 5>().data());

  }
private:
  TF_DISALLOW_COPY_AND_ASSIGN(SparsePadOp);
};

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("SparsePad") \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<TYPE>("dtype"),       \
                              SparsePadOp<Eigen::ThreadPoolDevice, TYPE>)

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("SparsePad") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          SparsePadOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

template <typename Device, typename T>
class SparsePadGradOp : public OpKernel {
 private:
  std::vector<int> strides_;
  std::vector<int> dilations_;
 public:
  explicit SparsePadGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES(ctx, strides_.size() == 3, errors::InvalidArgument("strides must be a vector have 3 element"));
    OP_REQUIRES(ctx, dilations_.size() == 3, errors::InvalidArgument("dilations must be a vector have 3 element"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& images = ctx->input(0);
    const Tensor& base_plane = ctx->input(1);
    const Tensor& out_grad = ctx->input(2);

    const auto batch_size = images.dim_size(0);
    const auto image_height = images.dim_size(1);
    const auto image_width = images.dim_size(2);
    const auto image_depth = images.dim_size(3);
    const auto image_channels = images.dim_size(4);

    const auto out_height = image_height * strides_[0];
    const auto out_width = image_width * strides_[1];
    const auto out_depth = image_depth * strides_[2];

    OP_REQUIRES(ctx, (base_plane.shape().dims() == 4) && (base_plane.dim_size(0) == batch_size)
                      && (base_plane.dim_size(1) == out_height) && (base_plane.dim_size(2) == out_width) && (base_plane.dim_size(3) == 1),
                errors::InvalidArgument("base_plane must have rank 4, and must compate to ref_image, got ", base_plane.shape().DebugString()));

    OP_REQUIRES(ctx, (out_grad.shape().dims() == 5) && (out_grad.dim_size(0) == batch_size) && (out_grad.dim_size(1) == out_height) 
                        && (out_grad.dim_size(2) == out_width) && (out_grad.dim_size(3) == out_depth) && (out_grad.dim_size(4) == image_channels),
                errors::InvalidArgument("out_grad must have rank 5, and must compate to ref_image, got ", out_grad.shape().DebugString()));

    Tensor *image_grad;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            images.shape(),
                            &image_grad));


    SparsePadGradFunctor<Device, T>()(ctx, ctx->eigen_device<Device>(),
                                      strides_[0],
                                      strides_[1],
                                      strides_[2],
                                      batch_size,
                                      image_height,
                                      image_width,
                                      image_depth,
                                      image_channels,
                                      out_height,
                                      out_width,
                                      out_depth,
                                      images.tensor<T, 5>().data(),
                                      base_plane.tensor<int32, 4>().data(),
                                      out_grad.tensor<T, 5>().data(),
                                      image_grad->tensor<T, 5>().data());

  }
private:
  TF_DISALLOW_COPY_AND_ASSIGN(SparsePadGradOp);
};

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("SparsePadGrad") \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<TYPE>("dtype"),       \
                              SparsePadGradOp<Eigen::ThreadPoolDevice, TYPE>)

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("SparsePadGrad") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),        \
                          SparsePadGradOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

} /* custom_helper_op */
} /* tensorflow */