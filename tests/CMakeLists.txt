
file(GLOB TESTFILES "*.c")

foreach(TESTFILE ${TESTFILES})

    get_filename_component(TESTEXEC ${TESTFILE} NAME_WE)

    add_executable(${TESTEXEC} ${TESTFILE})
    target_link_libraries(${TESTEXEC} PRIVATE scorch m)

    add_test(NAME ${TESTEXEC} COMMAND ${TESTEXEC})

endforeach()

