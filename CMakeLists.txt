cmake_minimum_required(VERSION 4.1)
project(simple_knn_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(simple_knn_cuda main.cu)

target_include_directories(my_simple_knn PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(my_simple_knn PRIVATE
        CUDA::cudart
)

set_target_properties(simple_knn_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(my_simple_knn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)