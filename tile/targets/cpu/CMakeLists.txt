cmake_minimum_required(VERSION 3.0)
project(tile)

jsonnet_to_json(OUT_FILE_NAMES llvm_cpu.json llvm_cpu.jsonnet)
heatmap(CPU_HEATMAP_CPP heatmap.cc heatmap_skx_xeonplat_8180_1-7GHz_mblocked.csv.gz heatmap.tpl.cc)


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
        ${OUT_FILE_NAMES}
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

