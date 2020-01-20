cmake_minimum_required(VERSION 3.0)

project(plaidml)


set_property(GLOBAL PROPERTY USE_FOLDERS On) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL MinSizeRel Release)
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO RelWithDebInfo Release)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

#for base
find_package(gflags REQUIRED CONFIG)
find_package(protobuf REQUIRED CONFIG)


#generate protobuf stuff

find_package(protobuf REQUIRED CONFIG)


set(PROTO_SRCS_SCHEMA
    tile/proto/hal.proto
    tile/proto/metadata.proto
    tile/proto/schedule.proto
    tile/proto/shape.proto
    tile/proto/tile.proto
    tile/lang/lang.proto
)

#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SRCS_SCHEMA})

set(LIB_NAME proto_lib)
add_library(${LIB_NAME} ${PROTO_SRCS} ${PROTO_HDRS})
add_library(plaidml::${LIB_NAME} ALIAS ${LIB_NAME})

set(protobuf_generate_PROTOC_OUT_DIR ${CMAKE_BINARY_DIR})
protobuf_generate(
 LANGUAGE cpp
 TARGET ${LIB_NAME}
 PROTOS ${PROTO_SRCS_SCHEMA})

target_link_libraries(${LIB_NAME} PUBLIC protobuf::libprotobuf)
target_include_directories(${LIB_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include/paidml>
)

set_target_properties(${LIB_NAME} PROPERTIES FOLDER base)

if (BUILD_SHARED_LIBS)
    set_target_properties(${LIB_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${LIB_NAME} PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
endif()


#proto end




add_subdirectory(tile)
