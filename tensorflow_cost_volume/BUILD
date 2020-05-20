licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])
load(":tensorflow_cost_volume.bzl", "custom_op_library")

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

custom_op_library(
    name = "_cost_volume.so",
    srcs = [
        "cc/kernels/cost_volume_kernels.cc",
        "cc/kernels/cost_volume.h",
        "cc/ops/cost_volume_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/cost_volume.h",
        "cc/kernels/cost_volume_kernels_gpu.cu.cc",
    ],
)

py_library(
    name = "tensorflow_cost_volume",
    srcs = glob(["*.py"]),
    data = [
        ":_cost_volume.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "time_two_ops_py_test",
    srcs = [
        "python/ops/time_two_ops_test.py"
    ],
    main = "python/ops/time_two_ops_test.py",
    deps = [
        ":time_two_ops_py",
    ],
    srcs_version = "PY2AND3",
)
