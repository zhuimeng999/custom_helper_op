#include "tensorflow/core/framework/op_kernel.h"
#include <bit>

void swap_endianess(uint32_t& ui)
{
  static const uint32_t A(0x000000ffU);
  static const uint32_t B(0x0000ff00U);
  static const uint32_t C(0x00ff0000U);
  static const uint32_t D(0xff000000U);

  ui = ( (ui & A) << 24 )
     | ( (ui & B) <<  8 )
     | ( (ui & C) >>  8 )
     | ( (ui & D) >> 24 );
}

namespace tensorflow {
namespace custom_helper_op {
namespace {
class DecodePFMOp : public OpKernel {
 public:
  explicit DecodePFMOp(OpKernelConstruction* context) : OpKernel(context) {
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    big =( bint.c[0] == 1); 
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const string input = input_tensor->scalar<tstring>()();
    size_t pos = 0;
    size_t off = input.find_first_of(" \t\r\n", pos);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no magic"));
    string magic = input.substr(pos, off - pos);
    OP_REQUIRES(
        context,
        (magic == "Pf" || magic == "PF"),
        errors::InvalidArgument("invalid format: ", magic));
    const int64 channels = (magic == "Pf") ? 1 : 3;

    off = input.find_first_not_of(" \t\r\n", off);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no width"));
    if (input[off] == '#') {
      // comment
      while (off < input.size() && input[off] != '\n') {
        off++;
      }
    }
    // width, height, max
    pos = off;
    off = input.find_first_of(" \t\r\n", pos);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no width"));
    int64 width;
    OP_REQUIRES(context,
                (strings::safe_strto64(input.substr(pos, off - pos), &width)),
                errors::InvalidArgument("unable to parse width: ",
                                        input.substr(pos, off - pos)));

    off = input.find_first_not_of(" \t\r\n", off);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no height"));
    pos = off;
    off = input.find_first_of(" \t\r\n", pos);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no height"));
    int64 height;
    OP_REQUIRES(context,
                (strings::safe_strto64(input.substr(pos, off - pos), &height)),
                errors::InvalidArgument("unable to parse height: ",
                                        input.substr(pos, off - pos)));

    off = input.find_first_not_of(" \t\r\n", off);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no scaling_factor"));
    pos = off;
    off = input.find_first_of(" \t\r\n", pos);
    OP_REQUIRES(context, (off != string::npos),
                errors::InvalidArgument("no scaling_factor"));
    float scaling_factor;
    OP_REQUIRES(context,
                (strings::safe_strtof(input.substr(pos, off - pos), &scaling_factor)),
                errors::InvalidArgument("unable to parse scaling_factor: ",
                                        input.substr(pos, off - pos)));
    bool endian_mismatched = big ^ (scaling_factor > 0.0f);
    OP_REQUIRES(context, !endian_mismatched,
                errors::InvalidArgument("invalid scaling_factor value: mismatched endian type ", scaling_factor, " and ", big));

    Tensor* image_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(
                     0, TensorShape({height, width, channels}), &image_tensor));
    off = off + 1;
    if (magic == "Pf") {
        OP_REQUIRES(context,
                    (off + image_tensor->NumElements()*4 <= input.size()),
                    errors::InvalidArgument("not enough data"));
        float* dataptr = image_tensor->flat<float>().data();
        const float* inputptr = (const float *)(&input.data()[off]);
        const size_t stepsize = width*4;
        for (auto h = 0; h < height; h++) { 
          memcpy(&dataptr[h*width], &inputptr[(height - 1 - h)*width], stepsize);
        }
    } else {
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "no implement"));
    }
  }

  bool big;
};
REGISTER_KERNEL_BUILDER(Name("DecodePfm").Device(DEVICE_CPU), DecodePFMOp);

}  // namespace
}  // namespace io
}  // namespace tensorflow