cmake_minimum_required(VERSION 3.0)
project(proto)

set(SRCS
    lib.h
    lib.cc
    tests.h
    tests.cc
)

paidml_add_library(tile_lib
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::lang
        plaidml::tile_util
        #//plaidml2/edsl:edsl_mli
        Boost::boost
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