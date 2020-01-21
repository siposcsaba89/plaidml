cmake_minimum_required(VERSION 3.0)
project(proto)


set(PUBLIC_HDRS
    basis.h
    bignum.h
    matrix.h
    polynomial.h
    util.h
)

set(SRCS
    basis.cc
    bignum.cc
    matrix.cc
    polynomial.cc
    util.cc
)

paidml_add_library(math
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::util
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