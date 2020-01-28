cmake_minimum_required(VERSION 3.0)
project(proto)


set(PUBLIC_HDRS
    enums.h
    interfaces.h
    slice.h
    util.h
)

set(SRCS
    enums.cc
    interfaces.cc
    slice.cc
    util.cc
)

set(LLVM_TARGET_DEFINITIONS enums.td)
mlir_tablegen(enums.h.inc -gen-enum-decls)
mlir_tablegen(enums.cc.inc -gen-enum-defs)
add_public_tablegen_target(pmlc_enums_tbl)

set(LLVM_TARGET_DEFINITIONS interfaces.td)
mlir_tablegen(interfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(interfaces.cc.inc -gen-op-interface-defs)
add_public_tablegen_target(pmlc_interfaces_tbl)

paidml_add_library(pmlc_util
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
        ${TABLEGEN_OUTPUT}
    FOLDER
        pmlc
    PUBLIC_DEPS
        plaidml::util
        plaidml::base
        LLVMSupport
        MLIRIR
    PRIVATE_DEFS
        -DNOMINMAX
)
set(TABLEGEN_OUTPUT )
target_include_directories(pmlc_util PUBLIC ${LLVM_INCLUDE_DIRS})
add_dependencies(pmlc_util pmlc_enums_tbl pmlc_interfaces_tbl)


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