set(HEADER_LIST "${Scorch_SOURCE_DIR}/include/scorch/scorch.h")

add_library(scorch ops.c graph.c scorch.c ${HEADER_LIST})

target_include_directories(scorch PUBLIC ../include)

target_link_libraries(scorch PRIVATE m)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})

clangformat_setup(
    graph.c
    graph.h
    ops.c
    ops.h
    scorch.c
    )
