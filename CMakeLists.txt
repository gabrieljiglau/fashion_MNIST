cmake_minimum_required(VERSION 4.0 FATAL_ERROR)
project(fashion_MNIST)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "$ENV{HOME}/libs/libtorch")

find_package(Torch REQUIRED)

add_executable(fashion_MNIST src/custom_network/main.cpp)
target_link_libraries(fashion_MNIST "${TORCH_LIBRARIES}")
set_property(TARGET fashion_MNIST PROPERTY CXX_STANDARD 17)