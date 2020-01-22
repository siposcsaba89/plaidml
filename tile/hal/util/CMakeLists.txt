cmake_minimum_required(VERSION 3.0)
project(proto)


paidml_add_library(selector
    PUBLIC_HEADERS
        selector.h
    SRCS
        selector.cc
    FOLDER
        tile/hal/util
    PUBLIC_DEPS
        Boost::regex
    PRIVATE_DEFS
        -DNOMINMAX
)


paidml_add_library(settings
    PUBLIC_HEADERS
        settings.h
    SRCS
        settings.cc
    FOLDER
        tile/hal/util
    PUBLIC_DEPS
        plaidml::lang
    PRIVATE_DEFS
        -DNOMINMAX
)
