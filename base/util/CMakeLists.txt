cmake_minimum_required(VERSION 3.0)

project(util)

set(PUBLIC_HDRS
    any_factory.h
    any_factory_map.h
    callback_map.h
    catch.h
    compat.h
    env.h
    error.h
    factory.h
    file.h
    hexdump.h
    intern.h
    iterator_util.h
    json_transfer.h
    logging.h
    lookup.h
    pdebug.h
    perf_counter.h
    stream_container.h
    sync.h
    throw.h
    transfer_object.h
    type_url.h
    uuid.h
    zipfile.h
)

set(SRCS
    env.cc
    error.cc
    file.cc
    hexdump.cc
    json_transfer.cc
    logging.cc
    perf_counter.cc
    uuid.cc
    zipfile.cc
    ${EASYLOGGINGPP_SRC}/easylogging++.cc
)

paidml_add_library(util
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        base
    PUBLIC_DEPS
        Boost::boost
        Boost::filesystem
        Boost::disable_autolinking
        #Boost::stacktrace
        gflags_static
        jsoncpp_lib
        MINIZIP::minizip
)
target_compile_definitions(util PRIVATE -DNOMINMAX)


paidml_add_library(runfiles_db
    PUBLIC_HEADERS
        runfiles_db.h
    SRCS
        runfiles_db.cc
    FOLDER
        base
    PUBLIC_DEPS
        plaidml::util
)
