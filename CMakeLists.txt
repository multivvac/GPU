cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda-challenge)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cuda-challenge main.cpp)
target_link_libraries(cuda-challenge "${TORCH_LIBRARIES}")
target_include_directories(cuda-challenge PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_property(TARGET cuda-challenge PROPERTY CXX_STANDARD 17)

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
