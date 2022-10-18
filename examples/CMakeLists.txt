
add_executable(iris_log_reg  log_reg.c)
configure_file(iris.csv iris.csv COPYONLY)

target_link_libraries(iris_log_reg PRIVATE scorch m)
