cmake_minimum_required(VERSION 3.0)
project(proto)


set(PUBLIC_HDRS
    compile.h
    compose.h
    emitc.h
    generate.h
    ops.h
    parser.h
    runinfo.h
    sembuilder.h
    semprinter.h
    semtree.h
    simplifier.h
    symbolic.h
    type.h
)

set(SRCS
    bound.cc
    bound.h
    builtins.cc
    builtins.h
    compile.cc
    compose.cc
    defract.cc
    defract.h
    emitc.cc
    exprtype.cc
    exprtype.h
    flat.cc
    flat.h
    fnv1a64.h
    fpconv.cc
    fpconv.h
    gen_contract.cc
    gen_contract.h
    gen_special.cc
    gen_special.h
    gen_stripe.cc
    gen_stripe.h
    gen_trivial.cc
    gen_trivial.h
    generate.cc
    gid.cc
    gid.h
    intrinsic.cc
    intrinsic.h
    loop.cc
    loop.h
    ops.cc
    out_plan.cc
    out_plan.h
    parser.cc
    read_plan.cc
    read_plan.h
    reduce.cc
    reduce.h
    replace.cc
    replace.h
    scope.h
    semprinter.cc
    semtree.cc
    simplifier.cc
    sym_poly.cc
    sym_poly.h
    symbolic.cc
    tile_cache.cc
    tile_cache.h
    tile_opt.cc
    tile_opt.h
    type.cc
    usedef.cc
    usedef.h
    #:parser
    #:parser_lex
)

find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
bison_target(MyParser tile.y ${CMAKE_CURRENT_BINARY_DIR}/parser.y.cpp
             DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/parser.y.h)
flex_target(MyScanner tile.lex  ${CMAKE_CURRENT_BINARY_DIR}/parser_lex.h)
add_flex_bison_dependency(MyScanner MyParser)
paidml_add_library(lang
    PUBLIC_HEADERS
        ${PUBLIC_HDRS}
    SRCS
        ${SRCS}
        ${BISON_MyParser_OUTPUTS}
        ${FLEX_MyScanner_OUTPUTS}
    FOLDER
        tile
    PUBLIC_DEPS
        plaidml::util
        plaidml::base
        plaidml::bilp
        plaidml::math
        plaidml::stripe
        Boost::boost
    PRIVATE_DEFS
        -DNOMINMAX
        -DYY_NO_UNISTD_H
)

#plaidml_cc_test(
#    name = "test",
#    srcs = [
#        "lang_test.cc",
#        "sim_test.cc",
#        "simulate.h",
#        "test.cc",
#    ],
#    tags = ["windows_fail"],
#    deps = [
#        ":lang",
#    ],
#)
#
#plaidml_bison(
#    name = "parser",
#    src = "tile.y",
#)
#
#plaidml_flex(
#    name = "parser_lex",
#    src = "tile.lex",
#)
#
#plaidml_cc_test(
#    name = "gid_test",
#    srcs = ["gid_test.cc"],
#    deps = [
#        ":lang",
#        "//base/util",
#    ],
#)
#
#plaidml_cc_test(
#    name = "simplifier_test",
#    srcs = ["simplifier_test.cc"],
#    deps = [":lang"],
#)
#
#plaidml_cc_test(
#    name = "gen_test",
#    srcs = ["gen_test.cc"],
#    tags = ["manual"],
#    deps = [":lang"],
#)
#
#plaidml_cc_test(
#    name = "exprtype_test",
#    srcs = ["exprtype_test.cc"],
#    deps = [":lang"],
#)
#
#plaidml_cc_test(
#    name = "gen_stripe_test",
#    srcs = ["gen_stripe_test.cc"],
#    tags = ["manual"],
#    deps = [
#        ":lang",
#        "//plaidml2/edsl:edsl_ast",
#        "//testing:matchers",
#    ],
#)
