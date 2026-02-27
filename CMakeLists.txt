cmake_minimum_required(VERSION 4.1)
project(simple_knn_cuda CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(simple_knn_cuda
        main.cu
        knn_cpu.cpp
        knn_cpu.h
        knn_cpu.h
        knn_cpu.cpp)

target_include_directories(simple_knn_cuda PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(simple_knn_cuda PRIVATE
        CUDA::cudart
)

set_target_properties(simple_knn_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(simple_knn_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
# Show warnings on host code
if (MSVC)
    target_compile_options(simple_knn_cuda PRIVATE /w4)
else()
    target_compile_options(simple_knn_cuda PRIVATE -Wall -Wextra -Wpedantic)
endif()
#Generate debug info for device in Debug
target_compile_options(simple_knn_cuda PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
)

# Compile for GPU architecture (Using RTS 2080 => SM75)
set_property(TARGET simple_knn_cuda PROPERTY CUDA_ARCHITECTURES 75)