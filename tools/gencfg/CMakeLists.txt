project(gencfg)

add_executable(gencfg gencfg.cc)
target_link_libraries(gencfg PRIVATE
    plaidml::config
    plaidml::util
    plaidml::codegen_proto_lib
    plaidml::stripe_proto_lib
    plaidml::proto_lib
    Boost::program_options
    Boost::disable_autolinking
)
target_compile_definitions(gencfg PRIVATE -DBOOST_PROGRAM_OPTIONS_DYN_LINK=1)