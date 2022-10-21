# Storch ![master](https://github.com/riccardosven/storch/actions/workflows/cmake.yml/badge.svg)

Storch is a very simple torch-like tensor operation engine.

It includes tensor manipulation functions and graph operations for autodifferentiation; the rationale is a torch-like lazy-forward lazy-backward evaluation of a dynamic computational graph.

## Build the project
```
 $ cmake -B build .
 $ cmake --build build
```

## Run unit and integration tests
```
 $ ctest --test-dir build
```

## Build documentation
```
 $ doxygen
```

## Requirements
 - cmake
 - check
 - openblas
 - doxygen (to build documentation)
