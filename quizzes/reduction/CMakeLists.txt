add_executable(reduction quiz.cpp reduction_kernel.cu)
target_link_libraries(reduction PRIVATE CUDA::cudart)
target_link_libraries(reduction PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(reduction PRIVATE ${CMAKE_SOURCE_DIR}/include)
