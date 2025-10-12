cmake_minimum_required(VERSION 3.10)
project(FastMatrix LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable better optimization flags when building Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Add executable
add_executable(fastmatrix
    src/mat_ops.cu
    src/mat_ops_cpu.cpp
    src/benchmark.cpp
)

# For CUDA separable compilation
set_target_properties(fastmatrix PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries if needed (none required for now)
