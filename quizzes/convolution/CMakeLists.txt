add_executable(convolution quiz.cpp convolution_kernel.cu)
target_link_libraries(convolution PRIVATE CUDA::cudart)
target_link_libraries(convolution PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(convolution PRIVATE ${CMAKE_SOURCE_DIR}/include)
