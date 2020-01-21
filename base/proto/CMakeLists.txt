
set(PUBLIC_HDRS
    proto.h
)

set(SRCS
    proto.cc
)

paidml_add_library(proto
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
    FOLDER
        base
    PUBLIC_DEPS
        plaidml::util
        protobuf::libprotobuf
)

