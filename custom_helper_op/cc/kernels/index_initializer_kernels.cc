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
#include "custom_helper_op/cc/kernels/index_initializer.h"

namespace tensorflow {
namespace custom_helper_op {

namespace functor {

template <typename T, bool half_centor>
struct FillIndexFunctor<Eigen::ThreadPoolDevice, T, half_centor> {
  void operator()(OpKernelContext* ctx, const Eigen::ThreadPoolDevice& d, T *out_data,int32 out_height, int32 out_width)
{
    auto initializer_function = [&](const int start, const int limit) {
      for (int i = start; i < limit; ++i) {
        T *tmp = &out_data[i*3];
        tmp[0] = T(i%out_width) + 0.5*half_centor;
        tmp[1] = T(i/out_width) + 0.5*half_centor;
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
template struct FillIndexFunctor<Eigen::ThreadPoolDevice, float, true>;
template struct FillIndexFunctor<Eigen::ThreadPoolDevice, float, false>;
template struct FillIndexFunctor<Eigen::ThreadPoolDevice, double, true>;
template struct FillIndexFunctor<Eigen::ThreadPoolDevice, double, false>;
} /* functor */

using functor::FillIndexFunctor;

template <typename Device, typename T>
class IndexInitializerOp : public OpKernel {
 private:
  bool half_centor_;
 public:
  explicit IndexInitializerOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("half_centor", &half_centor_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);

    OP_REQUIRES(ctx, shape_t.dtype() == DT_INT32,
      errors::InvalidArgument(
          "Bad size input type for SetOutputToSizedImage: Expected DT_INT32 "
          "but got ",
          DataTypeString(shape_t.dtype())));
    OP_REQUIRES(ctx, shape_t.dims() == 1,
                  errors::InvalidArgument("output shape must be 1-dimensional",
                                          shape_t.shape().DebugString()));
    OP_REQUIRES(ctx, shape_t.NumElements() == 3,
                  errors::InvalidArgument("output shape must have 3 elements",
                                          shape_t.shape().DebugString()));

    auto shape_vec = shape_t.flat<int32>();                    
    auto out_height = shape_vec(0);
    auto out_width = shape_vec(1);

    OP_REQUIRES(ctx, shape_vec(2) == 3,
                  errors::InvalidArgument("last dimsion must be 3",
                                          shape_vec(2)));

    Tensor *output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            TensorShape({out_height, out_width, 3}),
                            &output));


    if(half_centor_){
      FillIndexFunctor<Device, T, true>()(ctx, 
          ctx->eigen_device<Device>(), output->tensor<T, 3>().data(), out_height, out_width);
    } else {
      FillIndexFunctor<Device, T, false>()(ctx, 
          ctx->eigen_device<Device>(), output->tensor<T, 3>().data(), out_height, out_width);
    }
  }
};

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("IndexInitializer") \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<TYPE>("dtype"),       \
                              IndexInitializerOp<Eigen::ThreadPoolDevice, TYPE>)

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("IndexInitializer") \
                              .Device(DEVICE_GPU)                   \
                              .HostMemory("output_shape")            \
                              .TypeConstraint<TYPE>("dtype"),        \
                          IndexInitializerOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

} /* custom_helper_op */
} /* tensorflow */