
set(PUBLIC_HDRS
    factory.h
)

set(SRCS
    eventlog.cc
    eventlog.h
    factory.cc
)

paidml_add_library(file
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        base
    PUBLIC_DEPS
        plaidml::util
        plaidml::context
        protobuf::libprotobuf
)

#plaidml_cc_test(
#    name = "eventlog_test",
#    srcs = ["eventlog_test.cc"],
#    deps = [
#        ":file",
#        "//testing:matchers",
#    ],
#)