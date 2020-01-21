cmake_minimum_required(VERSION 3.0)

project(proto)


set(PUBLIC_HDRS
    stripe.h
)

set(SRCS
    impl.h
    stripe.cc
    protobuf.cc
)

paidml_add_library(stripe
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::stripe_proto_lib
        plaidml::util
        plaidml::base
        plaidml::math
        Boost::boost
        Boost::filesystem
)


#plaidml_cc_test(
#    name = "stripe_test",
#    srcs = ["stripe_test.cc"],
#    deps = [
#        ":stripe",
#        "//base/util",
#    ],
#)