cmake_minimum_required(VERSION 3.0)

project(proto)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(PUBLIC_HDRS
    support.h
)

set(SRCS
    support.cc
)

set(LIB_NAME support)

add_library(${LIB_NAME} ${PUBLIC_HDRS} ${SRCS})
add_library(plaidml::${LIB_NAME} ALIAS ${LIB_NAME})

target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include/plaidml>
)

target_link_libraries(${LIB_NAME} PUBLIC plaidml::proto_lib plaidml::base)

set_target_properties(${LIB_NAME} PROPERTIES FOLDER base)

if (BUILD_SHARED_LIBS)
    set_target_properties(${LIB_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${LIB_NAME} PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
endif()

if (MSVC)
    target_compile_options(${LIB_NAME} PRIVATE "/MP")
endif()

