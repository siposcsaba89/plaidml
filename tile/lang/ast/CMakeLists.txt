cmake_minimum_required(VERSION 3.0)
project(proto)


set(PUBLIC_HDRS
    ast.h
    gradient.h
)

set(SRCS
    ast.cc
    ast_ops.cc
    ast_ops.h
    fold.cc
    fold.h
    gradient.cc
    traversal.cc
    traversal.h
)

paidml_add_library(ast
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::util
        plaidml::base
        plaidml::lang
        Boost::boost
    PRIVATE_DEFS
        -DNOMINMAX
        -DBOOST_THREAD_PROVIDES_FUTURE

)