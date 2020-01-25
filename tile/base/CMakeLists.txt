cmake_minimum_required(VERSION 3.0)

project(base)

set(PUBLIC_HDRS
    buffer.h
    dbgsync.h
    lru_cache.h
    namespaces.h
    platform.h
    program.h
    shape.h
    validate.h
)


set(SRCS
    dbgsync.cc
    shape.cc
    validate.cc
)


paidml_add_library(base
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        tile
    PUBLIC_DEPS
        gflags_static
        plaidml::proto_lib
        plaidml::context
        plaidml::util
        Boost::boost
        Boost::thread
        Boost::disable_autolinking
)

paidml_add_library(schedule
    PUBLIC_HEADERS
        schedule.h
    SRCS
        schedule.cc
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::proto_lib
        plaidml::context
        plaidml::util
)

paidml_add_library(program_cache
    PUBLIC_HEADERS
        program_cache.h
    SRCS
        program_cache.cc
    FOLDER
        tile
    PUBLIC_DEPS
        Boost::boost
        Boost::disable_autolinking
        plaidml::base
        plaidml::support
        plaidml::stripe_proto_lib
        #plaidml::lang
    PRIVATE_DEFS
        -DNOMINMAX
        -DBOOST_THREAD_PROVIDES_FUTURE
)

#needs adding test
#plaidml_cc_library(
#    name = "platform_test",
#    testonly = True,
#    srcs = ["platform_test.cc"],
#    hdrs = ["platform_test.h"],
#    visibility = ["//visibility:public"],
#    deps = [
#        ":base",
#        "//testing:matchers",
#        "//tile/proto:support",
#        "@gmock//:gtest",
#        "@half",
#    ],
#    alwayslink = True,
#)
