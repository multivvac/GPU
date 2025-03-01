add_executable(histogram quiz.cpp solution.cpp)
target_link_libraries(histogram PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(histogram PRIVATE ${CMAKE_SOURCE_DIR}/include)
