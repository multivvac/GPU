add_executable(lenet main.cpp lenet_kernel.cu)
target_link_libraries(lenet PRIVATE CUDA::cudart)
target_link_libraries(lenet PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(lenet PRIVATE ${CMAKE_SOURCE_DIR}/include)
