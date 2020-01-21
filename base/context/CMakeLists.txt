
set(PUBLIC_HDRS
    context.h
    eventlog.h
    gate.h
)

set(SRCS
    context.cc
    eventlog.cc
    gate.cc
)

paidml_add_library(context
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        base
    PUBLIC_DEPS
        plaidml::util
        plaidml::context_proto_lib
        Boost::boost
        Boost::thread
)


#plaidml_cc_test(
#    name = "context_test",
#    srcs = ["context_test.cc"],
#    deps = [
#        ":context",
#        "//base/util",
#    ],
#)
#
#plaidml_cc_test(
#    name = "gate_test",
#    srcs = ["gate_test.cc"],
#    deps = [
#        ":context",
#        "//base/util",
#    ],
#)
