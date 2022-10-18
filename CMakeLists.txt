cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_STANDARD 11)
add_compile_options(-Wall -Wextra -pedantic -Werror)

project(Scorch
    VERSION 0.0.1
    DESCRIPTION "A torch clone"
    LANGUAGES C)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ClangFormat)

set(CB_VENDOR OpenBLAS)
find_package(BLAS)

include_directories("include")

add_subdirectory(src)

enable_testing()

add_subdirectory(tests)

