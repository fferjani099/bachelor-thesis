load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//tools/install:install.bzl", "install")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "traffic_prediction_lib",
    srcs = ["traffic_prediction.cc"],
    hdrs = ["traffic_prediction.h"],
    visibility = ["//visibility:private"],
    alwayslink = True,
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "@libtorch_gpu",
        "@local_config_cuda//cuda:cudart",
        "@opencv//:imgcodecs",
    ],
)

cc_binary(
    name = "lib_traffic_prediction.so",
    linkshared = True,
    linkstatic = True,
    deps = [":traffic_prediction_lib"],
)
cc_library(
    name = "depth_estimation_lib",
    srcs = ["depth_estimation.cc"],
    hdrs = ["depth_estimation.h"],
    visibility = ["//visibility:private"],
    alwayslink = True,
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
    ],
)
cc_binary(
    name = "lib_depth_estimation.so",
    linkshared = True,
    linkstatic = True,
    deps = [":depth_estimation_lib"],
)

cc_binary(
    name = "lib_object_tracking.so",
    linkshared = True,
    linkstatic = True,
    deps = [":object_tracking_lib"],
)

cc_library(
    name = "object_tracking_lib",
    srcs = ["object_tracking.cc"],
    hdrs = ["object_tracking.h"],
    visibility = ["//visibility:private"],
    alwayslink = True,
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
    ],
)
cc_binary(
    name = "lib_object_detection.so",
    linkshared = True,
    linkstatic = True,
    deps = [":object_detection_lib"],
)

cc_library(
    name = "object_detection_lib",
    srcs = ["object_detection.cc"],
    hdrs = ["object_detection.h"],
    visibility = ["//visibility:private"],
    alwayslink = True,
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
        "@libtorch_gpu",
        "@local_config_cuda//cuda:cudart",
        "@opencv//:imgcodecs",
    ],
)
cc_binary(
    name = "lib_pre_processing.so",
    linkshared = True,
    linkstatic = True,
    deps = [":pre_processing_lib"],
)

cc_library(
    name = "pre_processing_lib",
    srcs = ["pre_processing.cc"],
    hdrs = ["pre_processing.h"],
    visibility = ["//visibility:private"],
    alwayslink = True,
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",

    ],
)

cc_binary(
    name = "lib_camera2.so",
    linkshared = True,
    linkstatic = True,
    deps = [":camera2_lib"],
)

cc_library(
    name = "camera2_lib",
    srcs = ["camera2.cc"],
    hdrs = ["camera2.h"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
    ],
    alwayslink = True,
    visibility = ["//visibility:private"],
)
cc_binary(
    name = "lib_camera1.so",
    linkshared = True,
    linkstatic = True,
    deps = [":camera1_lib"],
)

cc_library(
    name = "camera1_lib",
    srcs = ["camera1.cc"],
    hdrs = ["camera1.h"],
    deps = [
        "//cyber",
        "//cyber/examples/proto:examples_cc_proto",
    ],
    alwayslink = True,
    visibility = ["//visibility:private"],
)
filegroup(
    name = "camera1_conf",
    srcs = [
        ":camera1.dag",
        ":case_study.launch",
    ],
)

install(
    name = "camera1_install",
    data = [
        ":camera1_conf",
    ],
    data_dest = "case_study/conf",
    library_dest = "case_study/lib",
    targets = [
        ":lib_camera1.so",
    ],
)
cpplint()
