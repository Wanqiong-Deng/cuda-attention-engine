cmake_minimum_required(VERSION 3.18)
project(cuda_attention LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include directories
include_directories(include)

# --- Library: attention kernels ---
add_library(attention_kernels
    src/naive_fp32.cu
    src/tiled_fp32.cu
    src/fp16.cu
    src/tensor_core.cu
    src/cpu_reference.cpp
)
target_include_directories(attention_kernels PUBLIC include)

# --- Test: correctness ---
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness attention_kernels)

# --- Benchmark ---
add_executable(benchmark bench/benchmark.cu)
target_link_libraries(benchmark attention_kernels)
