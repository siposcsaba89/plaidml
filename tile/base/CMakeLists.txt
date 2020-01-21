cmake_minimum_required(VERSION 3.0)

project(base)

find_package(gflags REQUIRED CONFIG)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND MSVC)
       set(Boost_COMPILER -vc141)
       #set(Boost_USE_DEBUG_RUNTIME OFF)
endif()
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.40 REQUIRED)


#find easylogging++
find_path(EASYLOGGINGPP_INCLUDE NAMES easylogging++.h)
find_path(EASYLOGGINGPP_SRC NAMES easylogging++.cc)

if (EASYLOGGINGPP_INCLUDE AND EASYLOGGINGPP_SRC)
    set(easyloggingpp_FOUND TRUE)
    message(STATUS "Found easylogging!")
else()
    set(easyloggingpp_FOUND FALSE)
    message(FATAL_ERROR "Not found easylogging!")
endif()

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
    ${EASYLOGGINGPP_SRC}/easylogging++.cc
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
