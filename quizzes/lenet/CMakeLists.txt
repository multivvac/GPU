set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -maxrregcount=32")
add_executable(lenet main.cpp lenet_kernel.cu)
target_link_libraries(lenet PRIVATE CUDA::cudart)
target_link_libraries(lenet PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(lenet PRIVATE ${CMAKE_SOURCE_DIR}/include)


add_executable(test_lenet test.cpp lenet_kernel.cu)
target_link_libraries(test_lenet PRIVATE CUDA::cudart "${TORCH_LIBRARIES}")
target_include_directories(test_lenet PRIVATE ${CMAKE_SOURCE_DIR}/include)
