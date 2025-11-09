cmake_minimum_required(VERSION 4.0 FATAL_ERROR)
project(fashion_MNIST)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "$ENV{HOME}/libs/eigen-3.4.0")
set(CMAKE_PREFIX_PATH "$ENV{HOME}/libs/libtorch")

find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(fashion_MNIST src/custom_network/main.cpp src/custom_network/activations.cpp)

if(TARGET Eigen3::Eigen)
    target_link_libraries(fashion_MNIST PRIVATE ${TORCH_LIBRARIES} Eigen3::Eigen)
else()
    include_directories(${EIGEN3_INCLUDE_DIR})
    target_link_libraries(fashion_MNIST "${TORCH_LIBRARIES}")
endif()

set_property(TARGET fashion_MNIST PROPERTY CXX_STANDARD 17)