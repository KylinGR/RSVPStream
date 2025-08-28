cmake_minimum_required(VERSION 3.10)  # 指定最低版本要求
project(rsvp_rknn)             # 项目名称

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)            
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#添加3rd目录
add_subdirectory(3rdparty/cnpy) #设置cnpy路径

# 头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rdparty/cnpy)

# 链接库路径
link_directories(${CMAKE_SOURCE_DIR}/lib)

# 链接的库
set(LINK_LIBS rknnrt)

# 多线程异步版本（主程序）
add_executable(rsvp_rknn_async
    src/main.cpp
    src/data_preprocessor.cpp
    src/inference_engine.cpp
    src/performance_evaluator.cpp
    src/async_pipeline.cpp
    src/utils.cpp
)

# 链接库到可执行文件
target_link_libraries(rsvp_rknn_async ${LINK_LIBS} cnpy pthread)

