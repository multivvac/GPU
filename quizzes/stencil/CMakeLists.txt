add_executable(stencil quiz.cpp stencil_kernel.cu)
target_link_libraries(stencil PRIVATE CUDA::cudart)
target_link_libraries(stencil PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(stencil PRIVATE ${CMAKE_SOURCE_DIR}/include)
