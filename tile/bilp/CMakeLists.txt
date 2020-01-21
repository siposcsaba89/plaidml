set(SRCS
    ilp_solver.cc
    ilp_solver.h
    tableau.cc
    tableau.h
)

paidml_add_library(bilp
    SRCS
        ${SRCS}
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::util
        plaidml::math
        GTest::gtest
    PRIVATE_DEFS
        -DNOMINMAX
)

#plaidml_cc_test(
#    name = "test",
#    srcs = [
#        "test.cc",
#    ],
#    # tags = ["large"],
#    timeout = "eternal",
#    deps = [
#        ":bilp",
#    ],
#)