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

using functor::FillIndexFunctor;

template <typename Device, typename T>
class SparseConv2DOp : public OpKernel {
 private:
  bool half_centor_;
 public:
  explicit SparseConv2DOp(OpKernelConstruction* ctx)
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

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("SparseConv2D") \
                              .Device(DEVICE_GPU)                   \
                              .HostMemory("output_shape")            \
                              .TypeConstraint<TYPE>("dtype"),        \
                          SparseConv2DOp<GPUDevice, TYPE>)

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

} /* custom_helper_op */
} /* tensorflow */