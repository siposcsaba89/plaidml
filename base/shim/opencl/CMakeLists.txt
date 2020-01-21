cmake_minimum_required(VERSION 3.0)

project(shim_opencl)
#TODO install location can be tricky needs to get in parameter

find_package(OpenCL REQUIRED)
add_library(::opencl SHARED IMPORTED)
set_property(TARGET ::opencl PROPERTY IMPORTED_IMPLIB_RELEASE ${OpenCL_LIBRARY})
set_property(TARGET ::opencl PROPERTY IMPORTED_LOCATION_RELEASE ${OpenCL_LIBRARY})
set_property(TARGET ::opencl PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_property(TARGET ::opencl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    $<INSTALL_INTERFACE:${OpenCL_INCLUDE_DIR}>  # <prefix>/include/mylib
    $<BUILD_INTERFACE:${OpenCL_INCLUDE_DIR}>  # <prefix>/include/mylib
)


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

