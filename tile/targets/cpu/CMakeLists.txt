cmake_minimum_required(VERSION 3.0)
project(tile)

set(CPU_LLVM_CPU_JSON_FILE ${CMAKE_CURRENT_BINARY_DIR}/llvm_cpu.json)
message(STATUS "Jsonnet executable : ${JSON_NET_EXECUTABLE}")
add_custom_command(OUTPUT ${CPU_LLVM_CPU_JSON_FILE}
  COMMAND ${JSON_NET_EXECUTABLE} ARGS -o ${CPU_LLVM_CPU_JSON_FILE} "${CMAKE_CURRENT_SOURCE_DIR}/llvm_cpu.jsonnet"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/llvm_cpu.jsonnet"
  COMMENT "jsonnet to json"
  VERBATIM
)

set(CPU_HEATMAP_CPP ${CMAKE_CURRENT_BINARY_DIR}/heatmap.cc)
add_custom_command(
        OUTPUT
            ${CPU_HEATMAP_CPP}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/heatmap_skx_xeonplat_8180_1-7GHz_mblocked.csv.gz ${CMAKE_CURRENT_SOURCE_DIR}/heatmap.tpl.cc
        COMMAND Python3::Interpreter "${CMAKE_SOURCE_DIR}/tools/heatmap/heatmap.py" "${CMAKE_CURRENT_SOURCE_DIR}/heatmap_skx_xeonplat_8180_1-7GHz_mblocked.csv.gz" "${CMAKE_CURRENT_SOURCE_DIR}/heatmap.tpl.cc" "${CPU_HEATMAP_CPP}"
        COMMENT Generating heatmap
)

paidml_add_library(cpu
    SRCS
        compiler.cc
        compiler.h
        config.h
        executable.cc
        executable.h
        heatmap.h
        jit.cc
        jit.h
        link_names.cc
        link_names.h
        programmodule.h
        ${CPU_LLVM_CPU_JSON_FILE}
        ${CPU_HEATMAP_CPP}
    FOLDER
        tile/targets
    PUBLIC_DEPS
        plaidml::stripe
        #half - copy include hpp
        TBB::tbb
        xsmm::xsmm
        LLVMExecutionEngine
        LLVMipo
        LLVMMCJIT
        LLVMX86AsmParser
        LLVMX86CodeGen
    PRIVATE_DEFS
        -DNOMINMAX
)

target_include_directories(cpu PUBLIC ${LLVM_INCLUDE_DIRS})

