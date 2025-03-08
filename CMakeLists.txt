cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda-challenge LANGUAGES CXX CUDA)

# If LibTorch is inside third_party/, set the prefix so CMake can find it.
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")

# Enable compile_commands.json (handy for IDEs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Let CMake handle the CUDA language
enable_language(CUDA)
set(CUDA_ARCHITECTURES 70)

# Find packages
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Debugging info
message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS = ${TORCH_INCLUDE_DIRS}")


add_executable(cuda-challenge
    src/main.cpp
)

#  The main executable also needs headers and Torch
target_include_directories(cuda-challenge
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)


target_link_libraries(cuda-challenge
    PRIVATE
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
)

# Use C++17
set_property(TARGET cuda-challenge PROPERTY CXX_STANDARD 17)

# Add quizzes subdirectory
add_subdirectory(quizzes)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET cuda-challenge
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:cuda-challenge>)
endif (MSVC)
