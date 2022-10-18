cmake_minimum_required(VERSION 3.10)


project(Scorch
    VERSION 0.0.1
    DESCRIPTION "A torch clone"
    LANGUAGES C)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ClangFormat)

include_directories("include")

add_subdirectory(src)

enable_testing()

add_subdirectory(tests)

