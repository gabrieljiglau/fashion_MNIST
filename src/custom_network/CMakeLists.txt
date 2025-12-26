add_library(custom_network
    network.cpp
    activations.cpp
    losses.cpp
    sweep.cpp
    ../utils.cpp) # relative to the current directory path

target_link_libraries(custom_network PUBLIC ${TORCH_LIBRARIES})
target_include_directories(custom_network PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})