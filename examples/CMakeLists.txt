
add_executable(iris_log_reg  log_reg.c)
configure_file(iris.csv iris.csv COPYONLY)

target_link_libraries(iris_log_reg PRIVATE storch)

add_executable(least_squares  least_squares.c)
target_link_libraries(least_squares PRIVATE storch)

add_executable(convex_2d  convex_2d.c)
target_link_libraries(convex_2d PRIVATE storch)
