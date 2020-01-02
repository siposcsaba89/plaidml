# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "COPTS")

plaidml_cc_library(
    name = "stripe_to_spirv",
    srcs = [
        "convert_stripe_to_spirv.cc",
        "convert_stripe_to_spirv.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        "//pmlc/dialect/stripe",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/stripe:passes",
    ],
    alwayslink = 1,
)
