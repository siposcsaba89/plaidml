cmake_minimum_required(VERSION 3.0)
project(proto)

set(SRCS
    emitsem.cc
    emitsem.h
    intrinsic.cc
    intrinsic.h
    kern_info.cc
    kern_info.h
    stripe_gen.cc
    stripe_gen.h
)

paidml_add_library(ocl_exec
    SRCS
        ${SRCS}
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::base
        plaidml::codegen
        plaidml::lang
        plaidml::targets
        plaidml::util
        Boost::filesystem
    PRIVATE_DEFS
        -DNOMINMAX
)


#plaidml_cc_test(
#    name = "test",
#    srcs = [
#        "math_test.cc",
#        "test.cc",
#    ],
#    tags = ["windows_fail"],
#    deps = [
#        ":math",
#    ],
#)