add_executable(histogram quiz.cpp histogram_kernel.cu)
target_link_libraries(histogram PRIVATE CUDA::cudart)
target_link_libraries(histogram PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(histogram PRIVATE ${CMAKE_SOURCE_DIR}/include)
