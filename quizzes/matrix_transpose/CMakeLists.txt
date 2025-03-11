add_executable(matrix_transpose quiz.cpp kernel.cu)
target_link_libraries(matrix_transpose PRIVATE CUDA::cudart)
target_link_libraries(matrix_transpose PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(matrix_transpose PRIVATE ${CMAKE_SOURCE_DIR}/include)
