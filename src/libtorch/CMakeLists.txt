add_library(libtorch
    torch_network.cpp)

target_link_libraries(libtorch PUBLIC ${TORCH_LIBRARIES})
target_include_directories(libtorch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})