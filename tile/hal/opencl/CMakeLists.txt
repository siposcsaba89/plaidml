cmake_minimum_required(VERSION 3.0)
project(proto)


paidml_add_library(opencl_hal
    PUBLIC_HEADERS
        emitocl.h
        info.h
    SRCS
        buffer.cc
        buffer.h
        cl_mem_arena.cc
        cl_mem_arena.h
        cl_mem_buffer.cc
        cl_mem_buffer.h
        cl_opt.cc
        cl_opt.h
        compiler.cc
        compiler.h
        compute_kernel.cc
        compute_kernel.h
        device.cc
        device.h
        device_memory.cc
        device_memory.h
        device_set.cc
        device_set.h
        device_state.cc
        device_state.h
        driver.cc
        driver.h
        emitocl.cc
        event.cc
        event.h
        executable.cc
        executable.h
        executor.cc
        executor.h
        host_memory.cc
        host_memory.h
        info.cc
        kernel.h
        library.cc
        library.h
        ocl.cc
        ocl.h
        opencl.cc
        result.cc
        result.h
        shared_memory.cc
        zero_kernel.cc
        zero_kernel.h
    FOLDER
        tile/hal/
    PUBLIC_DEPS
        plaidml::opencl_proto_lib
        plaidml::opencl
        plaidml::util
        plaidml::base
        plaidml::selector
        plaidml::lang
        plaidml::support
        Boost::filesystem
        Boost::regex
    PRIVATE_DEFS
        -DNOMINMAX
        -DBOOST_THREAD_PROVIDES_FUTURE
        -DBOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
        -D_WINSOCK_DEPRECATED_NO_WARNINGS
    PUBLIC_DEFS
        -D_WIN32_WINDOWS=0x0501
)


#plaidml_cc_test(
#    name = "platform_test",
#    srcs = ["platform_test.cc"],
#    tags = [
#        "manual",
#        "opencl",
#    ],
#    deps = [
#        ":opencl",
#        "//tile/base:platform_test",
#        "//tile/platform/local_machine",
#    ],
#)