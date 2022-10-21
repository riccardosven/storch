set(HEADER_LIST "${Storch_SOURCE_DIR}/include/storch/storch.h")

add_library(storch tensor.c arena.c ops.c graph.c storch.c ${HEADER_LIST})

target_include_directories(storch PUBLIC ../include)

target_link_libraries(storch PRIVATE m blas)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})

clangformat_setup(
    graph.c
    graph.h
    ops.c
    ops.h
    storch.c
    )
