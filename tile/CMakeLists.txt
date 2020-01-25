project(plaidml_tile)

add_subdirectory(proto)
add_subdirectory(base)
add_subdirectory(math)
add_subdirectory(stripe)
add_subdirectory(bilp)
add_subdirectory(lang)
add_subdirectory(lang/ast)


add_subdirectory(hal/util)
add_subdirectory(hal/opencl)

add_subdirectory(targets/cpu)


add_subdirectory(codegen)
add_subdirectory(ocl_exec)

