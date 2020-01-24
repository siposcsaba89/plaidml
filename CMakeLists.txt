cmake_minimum_required(VERSION 3.0)

project(plaidml)


set_property(GLOBAL PROPERTY USE_FOLDERS On) 

set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL MinSizeRel Release)
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO RelWithDebInfo Release)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

#for base
find_package(gflags REQUIRED CONFIG)
find_package(protobuf REQUIRED CONFIG)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND MSVC)
       set(Boost_COMPILER -vc141)
       #set(Boost_USE_DEBUG_RUNTIME OFF)
endif()
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.40 REQUIRED  COMPONENTS filesystem thread regex)

find_package(jsoncpp REQUIRED CONFIG)
#https://github.com/nmoinvaz/minizip fixing cmake config
find_package(minizip REQUIRED CONFIG)
find_package(GTest REQUIRED CONFIG)

find_package(LLVM REQUIRED CONFIG)

find_program (JSON_NET_EXECUTABLE jsonnet)

if (NOT JSON_NET_EXECUTABLE)
    message(FATAL_ERROR "Cannot find jsonnet")
else()
    message(STATUS "jsonnet executable found : ${JSON_NET_EXECUTABLE}")
endif()

find_package (Python3 COMPONENTS Interpreter REQUIRED)
message(STATUS "Python executable found: ${PYTHON_EXECUTABLE}")
find_package(TBB REQUIRED)
find_package(xsmm REQUIRED)


#generate protobuf stuff

function(paidml_add_library LIBRARY_NAME)
    set(options OPTIONAL)
    set(oneValueArgs DESTINATION RENAME FOLDER)
    set(multiValueArgs PUBLIC_HEADERS SRCS PUBLIC_DEPS PRIVATE_DEPS PUBLIC_DEFS PRIVATE_DEFS)
    cmake_parse_arguments(PARSE_ARGV 1
        "paidml_add_library" #prefix
        "${options}" #options
        "${oneValueArgs}" # one value arguments
        "${multiValueArgs}") # multi value arguments
    add_library(${LIBRARY_NAME}
        ${paidml_add_library_PUBLIC_HEADERS}
        ${paidml_add_library_SRCS})
    
    add_library(plaidml::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})
    
    target_include_directories(${LIBRARY_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<INSTALL_INTERFACE:include/plaidml>
        ${EASYLOGGINGPP_INCLUDE})
    target_link_libraries(${LIBRARY_NAME} PUBLIC ${paidml_add_library_PUBLIC_DEPS}
        PRIVATE ${paidml_add_library_PRIVATE_DEPS})
    set_target_properties(${LIBRARY_NAME} PROPERTIES 
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED TRUE)
    if (paidml_add_library_FOLDER)
        set_target_properties(${LIBRARY_NAME} PROPERTIES FOLDER ${paidml_add_library_FOLDER})
    endif()
    if (BUILD_SHARED_LIBS)
        set_target_properties(${LIBRARY_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
        set_target_properties(${LIBRARY_NAME} PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
    endif()

    if (MSVC)
        target_compile_options(${LIBRARY_NAME} PRIVATE "/MP")
    endif()

    if (paidml_add_library_PUBLIC_DEFS)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC ${paidml_add_library_PUBLIC_DEFS})
    endif()

    if (paidml_add_library_PRIVATE_DEFS)
        target_compile_definitions(${LIBRARY_NAME} PRIVATE ${paidml_add_library_PRIVATE_DEFS})
    endif()

endfunction()


#paidml_add_library(alma PUBLIC_HEADERS alma.h alma2.h alma3.h $<INSTALL_INTERFACE:${alma_SAdasd}> FOLDER kosar)


set(PROTO_SRCS_SCHEMA
    tile/proto/hal.proto
    tile/proto/metadata.proto
    tile/proto/schedule.proto
    tile/proto/shape.proto
    tile/proto/tile.proto
    tile/lang/lang.proto
)

#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SRCS_SCHEMA})

paidml_add_library(proto_lib FOLDER protobuf PUBLIC_DEPS protobuf::libprotobuf)
#set(protobuf_generate_PROTOC_OUT_DIR ${CMAKE_BINARY_DIR}/almasdasd)
protobuf_generate(
    LANGUAGE cpp
    TARGET proto_lib
    PROTOS ${PROTO_SRCS_SCHEMA})


paidml_add_library(context_proto_lib FOLDER protobuf PUBLIC_DEPS protobuf::libprotobuf)
protobuf_generate(
    LANGUAGE cpp
    TARGET context_proto_lib
    PROTOS base/context/context.proto)


paidml_add_library(file_proto_lib FOLDER protobuf PUBLIC_DEPS protobuf::libprotobuf)
protobuf_generate(
    LANGUAGE cpp
    TARGET file_proto_lib
    PROTOS base/eventing/file/eventlog.proto)

paidml_add_library(stripe_proto_lib FOLDER protobuf PUBLIC_DEPS protobuf::libprotobuf)
protobuf_generate(
    LANGUAGE cpp
    TARGET stripe_proto_lib
    PROTOS tile/stripe/stripe.proto)

paidml_add_library(codegen_proto_lib FOLDER protobuf PUBLIC_DEPS protobuf::libprotobuf)
protobuf_generate(
    LANGUAGE cpp
    TARGET codegen_proto_lib
    PROTOS tile/codegen/codegen.proto)


paidml_add_library(opencl_proto_lib FOLDER protobuf PUBLIC_DEPS protobuf::libprotobuf)
protobuf_generate(
    LANGUAGE cpp
    TARGET opencl_proto_lib
    PROTOS tile/hal/opencl/opencl.proto)


add_subdirectory(base)
add_subdirectory(tile)
