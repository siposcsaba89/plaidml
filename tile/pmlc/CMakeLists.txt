cmake_minimum_required(VERSION 3.0)
project(proto)


set(PUBLIC_HDRS
    pmlc.h
)

set(SRCS
    pmlc.cc
)

paidml_add_library(pmlc_lib
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::config
        plaidml::base
        plaidml::codegen
        plaidml::lang
        plaidml::tile_util
        plaidml::tile_lib
        Boost::program_options
    PRIVATE_DEFS
        -DNOMINMAX
    PUBLIC_DEFS
        -DBOOST_PROGRAM_OPTIONS_DYN_LINK=1
)

if (NOT MSVC)
    target_compile_definitions(pmlc PUBLIC -DENABLE_LLVM_BITCODE)
endif()


add_executable(pmlc main.cc)
target_link_libraries(pmlc PRIVATE plaidml::pmlc_lib)
set_target_properties(pmlc PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED TRUE)
target_include_directories(pmlc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include/plaidml>)



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