cmake_minimum_required(VERSION 3.16)
project(RSVPStream)

set(CMAKE_CXX_STANDARD 17)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 包含头文件
include_directories(include)

# 添加子目录
add_subdirectory(src)
add_subdirectory(tests)
