cmake_minimum_required(VERSION 3.0)

project(base)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL MinSizeRel Release)
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO RelWithDebInfo Release)


set(PUBLIC_HDRS
    buffer.h
    dbgsync.h
    lru_cache.h
    namespaces.h
    platform.h
    program.h
    shape.h
    validate.h
)


set(SRCS
    dbgsync.cc
    shape.cc
    validate.cc
)


find_package(gflags REQUIRED CONFIG)

add_library(base ${PUBLIC_HDRS} ${SRCS})
add_library(plaidml::base ALIAS base)

target_include_directories(base PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(base PUBLIC gflags_static)

set_target_properties(base PROPERTIES FOLDER base)

if (BUILD_SHARED_LIBS)
    set_target_properties(${PROJECT_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${PROJECT_NAME} PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
endif()

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE "/MP")
endif()

