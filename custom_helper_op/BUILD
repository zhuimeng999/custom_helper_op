licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])
load(":custom_helper_op.bzl", "custom_op_library")

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

custom_op_library(
    name = "_custom_helper_ops.so",
    srcs = [
        "cc/kernels/cost_volume_kernels.cc",
        "cc/kernels/cost_volume.h",

        "cc/kernels/cost_volume_kernels_v2.cc",
        "cc/kernels/cost_volume_v2.h",

        "cc/kernels/cost_volume_kernels_v3.cc",
        "cc/kernels/cost_volume_v3.h",

        "cc/kernels/cost_aggregate_kernels.cc",
        "cc/kernels/cost_aggregate.h",

        "cc/kernels/feature_aggregate_kernels.cc",
        "cc/kernels/feature_aggregate.h",

        "cc/kernels/sparse_conv_kernels.cc",
        "cc/kernels/sparse_conv.h",

        "cc/kernels/sparse_pad_kernels.cc",
        "cc/kernels/sparse_pad.h",

        "cc/kernels/deformable_conv_op.cc",
        "cc/kernels/deformable_conv_op.h",
        "cc/kernels/deformable_conv2d_utils.h",

        "cc/kernels/index_initializer_kernels.cc",
        "cc/kernels/index_initializer.h",

        "cc/kernels/pfm_decoder_kernels.cc",

        "cc/ops/custom_helper_ops.cc",
        "cc/ops/deformable_conv2d.cc"
    ],
    cuda_srcs = [
        "cc/kernels/cost_volume.h",
        "cc/kernels/cost_volume_kernels_gpu.cu.cc",

        "cc/kernels/cost_volume_v2.h",
        "cc/kernels/cost_volume_kernels_gpu_v2.cu.cc",

        "cc/kernels/cost_volume_v3.h",
        "cc/kernels/cost_volume_kernels_gpu_v3.cu.cc",

        "cc/kernels/cost_aggregate.h",
        "cc/kernels/cost_aggregate_kernels_gpu.cu.cc",

        "cc/kernels/feature_aggregate.h",
        "cc/kernels/feature_aggregate_kernels_gpu.cu.cc",

        "cc/kernels/sparse_conv.h",
        "cc/kernels/sparse_conv2d_kernels_gpu.cu.cc",
        "cc/kernels/sparse_conv3d_kernels_gpu.cu.cc",

        "cc/kernels/sparse_pad_kernels_gpu.cu.cc",
        "cc/kernels/sparse_pad.h",

        "cc/kernels/deformable_conv_op_gpu.cu.cc",
        "cc/kernels/deformable_conv_op.h",

        "cc/kernels/index_initializer.h",
        "cc/kernels/index_initializer_kernels_gpu.cu.cc",
    ],
)

py_library(
    name = "custom_helper_op",
    srcs = glob(["**/*.py"], exclude = ["python/ops/*_test.py"]),
    data = [
        ":_custom_helper_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "cost_volume_op_py_test",
    srcs = [
        "python/ops/op_tests/cost_volume_op_test.py"
    ],
    main = "python/ops/op_tests/cost_volume_op_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "cost_volume_v2_op_py_test",
    srcs = [
        "python/ops/op_tests/cost_volume_v2_op_test.py"
    ],
    main = "python/ops/op_tests/cost_volume_v2_op_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "cost_volume_v3_op_py_test",
    srcs = [
        "python/ops/op_tests/cost_volume_v3_op_test.py"
    ],
    main = "python/ops/op_tests/cost_volume_v3_op_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "cost_aggregate_op_py_test",
    srcs = [
        "python/ops/op_tests/cost_aggregate_op_test.py"
    ],
    main = "python/ops/op_tests/cost_aggregate_op_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "feature_aggregate_op_py_test",
    srcs = [
        "python/ops/op_tests/feature_aggregate_op_test.py"
    ],
    main = "python/ops/op_tests/feature_aggregate_op_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "sparse_conv_op_py_test",
    srcs = [
        "python/ops/op_tests/sparse_conv_op_test.py"
    ],
    main = "python/ops/op_tests/sparse_conv_op_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "sparse_pad_op_py_test",
    srcs = [
        "python/ops/op_tests/sparse_pad_op_test.py"
    ],
    main = "python/ops/op_tests/sparse_pad_op_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "index_initializer_op_py_test",
    srcs = [
        "python/ops/op_tests/index_initializer_op_test.py"
    ],
    main = "python/ops/op_tests/index_initializer_op_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)


py_test(
    name = "decoder_op_py_test",
    srcs = [
        "python/ops/op_tests/decoder_op_test.py"
    ],
    main = "python/ops/op_tests/decoder_op_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "depth_projection_layer_py_test",
    srcs = [
        "python/layers/depth_projection_layer_test.py"
    ],
    main = "python/layers/depth_projection_layer_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "deformable_conv2d_py_test",
    srcs = [
        "python/layers/deformable_conv2d_test.py"
    ],
    main = "python/layers/deformable_conv2d_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)
