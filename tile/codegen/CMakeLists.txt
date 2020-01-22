cmake_minimum_required(VERSION 3.0)
project(proto)

file(GLOB CODEGEN_SRCS *.cc *.h)


paidml_add_library(codegen
    SRCS
        ${CODEGEN_SRCS}
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::codegen_proto_lib
        plaidml::config
        plaidml::util
        plaidml::stripe
        plaidml::bilp
        #plaidml::cpu
        #plaidml::passes
        #plaidml::transcode
        Boost::filesystem

    PRIVATE_DEFS
        -DNOMINMAX
)


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