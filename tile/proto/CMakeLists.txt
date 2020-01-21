cmake_minimum_required(VERSION 3.0)

project(proto)


set(PUBLIC_HDRS
    support.h
)

set(SRCS
    support.cc
)

paidml_add_library(support
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::proto_lib
        plaidml::base
)
