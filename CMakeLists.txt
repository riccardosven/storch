cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_STANDARD 11)
add_compile_options(-Wall -Wextra -pedantic -Werror)

project(Storch
    VERSION 0.0.1
    DESCRIPTION "A simple implementation of a torch-like autodiff engine."
    LANGUAGES C)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ClangFormat)

find_package(CBLAS REQUIRED)
find_package(CHECK REQUIRED)

message("Found CBLAS libraries: " ${CBLAS_LIBRARIES})

include_directories("include")

add_subdirectory(src)

add_subdirectory(examples)

include(CTest)
enable_testing()

add_subdirectory(tests)

