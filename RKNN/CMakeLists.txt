cmake_minimum_required(VERSION 3.10)  # 指定最低版本要求
project(rknn_deploy)             # 项目名称

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)            
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 源文件
set(SOURCE_FILE1 ${CMAKE_SOURCE_DIR}/src/rknn_model.cpp) # 设置源文件路径
set(MAIN_FILE ${CMAKE_SOURCE_DIR}/src/rknn_main.cpp) # 设置源文件路径

# 头文件路径
include_directories(${CMAKE_SOURCE_DIR}/include)

# 链接库路径
link_directories(${CMAKE_SOURCE_DIR}/lib)

# 链接的库
set(LINK_LIBS rknnrt)

# 可执行文件输出名称
add_executable(rknn_main.out ${SOURCE_FILE1} ${MAIN_FILE})

# 链接库到可执行文件
target_link_libraries(rknn_main.out ${LINK_LIBS})
