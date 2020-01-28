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

set(LLVM_TARGET_DEFINITIONS ops.td)
mlir_tablegen(ops.h.inc -gen-op-decls)
mlir_tablegen(ops.cc.inc -gen-op-defs)
add_public_tablegen_target(pmlc_eltwise_ir_opt_tbl)

set(LLVM_TARGET_DEFINITIONS interfaces.td)
mlir_tablegen(interfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(interfaces.cc.inc -gen-op-interface-defs)
add_public_tablegen_target(pmlc_eltwise_ir_interfaces_tbl)

paidml_add_library(pmcl_eltwise_ir_impl
    PUBLIC_HEADERS
        dialect.h
        ops.h
        types.h
        util.h
    SRCS
        dialect.cc
        ops.cc
        types.cc
        util.cc
        ${TABLEGEN_OUTPUT}
    FOLDER
        pmlc/dialect/eltwise
    PUBLIC_DEPS
        plaidml::pmlc_util
        plaidml::base
        LLVMSupport
        MLIRIR
        MLIRStandardOps
    PRIVATE_DEFS
        -DNOMINMAX
)
set(TABLEGEN_OUTPUT )
target_include_directories(pmcl_eltwise_ir_impl PUBLIC ${LLVM_INCLUDE_DIRS})
add_dependencies(pmcl_eltwise_ir_impl pmlc_eltwise_ir_opt_tbl pmlc_eltwise_ir_interfaces_tbl)


paidml_add_library(pmcl_eltwise_ir
    SRCS
        registration.cc
    FOLDER
        pmlc/dialect/eltwise
    PUBLIC_DEPS
        pmcl_eltwise_ir_impl
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