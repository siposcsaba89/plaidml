cmake_minimum_required(VERSION 3.0)
project(tile)

jsonnet_to_json(OUT_FILE_NAMES_CPU llvm_cpu.json cpu/llvm_cpu.jsonnet)
jsonnet_to_json(OUT_FILE_NAMES_AMD amd.json gpu/amd.jsonnet)
jsonnet_to_json(OUT_FILE_NAMES_CM cm.json gpu/cm.jsonnet)
jsonnet_to_json(OUT_FILE_NAMES_INTEL_GEN9 intel_gen9.json gpu/intel_gen9.jsonnet)
jsonnet_to_json(OUT_FILE_NAMES_NVIDIA nvidia.json gpu/nvidia.jsonnet)

set(JSONS )

list(APPEND JSONS ${OUT_FILE_NAMES_CPU} ${OUT_FILE_NAMES_AMD} ${OUT_FILE_NAMES_CM} ${OUT_FILE_NAMES_INTEL_GEN9} ${OUT_FILE_NAMES_NVIDIA})

message(STATUS ${JSONS})
generate_config(OUT_CONFIG_NAME configs.h "vertexai::tile::targets::kConfigs"
    "${JSONS}"
)

paidml_add_library(targets
    PUBLIC_HEADERS
        targets.h
    SRCS
        targets.cc
        ${OUT_CONFIG_NAME}
    FOLDER
        tile/
    PUBLIC_DEPS
        plaidml::codegen
        plaidml::cpu
    PRIVATE_DEFS
        -DNOMINMAX
)

target_include_directories(cpu PUBLIC ${LLVM_INCLUDE_DIRS})

