cmake_minimum_required(VERSION 3.0)

project(config)


set(PUBLIC_HDRS
    config.h
)

set(SRCS
    config.cc
)

paidml_add_library(config
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        base
    PUBLIC_DEPS
        protobuf::libprotobuf
        plaidml::util
        plaidml::file_proto_lib #protobuf stuff
)