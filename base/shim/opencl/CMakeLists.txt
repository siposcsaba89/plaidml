cmake_minimum_required(VERSION 3.0)

project(shim_opencl)
#TODO install location can be tricky needs to get in parameter


set(PUBLIC_HDRS
    opencl.h
)

set(SRCS
    opencl.cc
)

paidml_add_library(opencl
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        base
    PUBLIC_DEPS
        plaidml::util
        ::opencl
    PUBLIC_DEFS
        -DCL_USE_DEPRECATED_OPENCL_1_2_APIS
        -DCL_SILENCE_DEPRECATION
)

