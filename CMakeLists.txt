cmake_minimum_required(VERSION 3.10)  # 指定最低版本要求
project(rsvp_rknn)             # 项目名称

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)            
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 检测是否为ARM架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8")
    message(STATUS "Detected ARM64 architecture, enabling NEON optimizations")
    
    # 为RK3588优化的编译选项
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+fp+simd")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=cortex-a76")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    
    # 可选：启用更激进的优化
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funsafe-math-optimizations")
    
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    message(STATUS "Detected ARM32 architecture, enabling NEON optimizations")
    
    # ARM32 NEON支持
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    
else()
    message(STATUS "Non-ARM architecture detected")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# 添加编译定义，确保代码知道NEON已启用
add_definitions(-D__ARM_NEON)

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

#所有cpp文件
file(GLOB SRC_FILES src/*.cpp)

# 可执行文件输出名称
add_executable(rsvp_rknn ${SRC_FILES})

# 链接库到可执行文件
target_link_libraries(rsvp_rknn ${LINK_LIBS} cnpy)

