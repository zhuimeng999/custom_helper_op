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
        "cc/kernels/index_initializer_kernels.cc",

        "cc/kernels/index_initializer.h",
        "cc/ops/custom_helper_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/cost_volume.h",
        "cc/kernels/cost_volume_kernels_gpu.cu.cc",

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
        "python/ops/cost_volume_op_test.py"
    ],
    main = "python/ops/cost_volume_op_test.py",
    deps = [
        ":custom_helper_op",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "index_initializer_op_py_test",
    srcs = [
        "python/ops/index_initializer_op_test.py"
    ],
    main = "python/ops/index_initializer_op_test.py",
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