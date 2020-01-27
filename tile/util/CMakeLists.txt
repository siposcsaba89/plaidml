cmake_minimum_required(VERSION 3.0)
project(proto)


set(PUBLIC_HDRS
    tile_file.h
)

set(SRCS
    tile_file.cc
)

paidml_add_library(tile_util
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::util
        plaidml::lang
        plaidml::proto_lib
        Boost::boost
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