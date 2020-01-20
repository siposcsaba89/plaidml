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



add_subdirectory(tile)
