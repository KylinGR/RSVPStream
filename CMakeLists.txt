cmake_minimum_required(VERSION 3.10)
project(rsvp_rknn)

set(CMAKE_CXX_STANDARD 17)            
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=cortex-a76 -ftree-vectorize")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXX flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CXX flags (Debug): ${CMAKE_CXX_FLAGS_DEBUG}")

add_subdirectory(3rdparty/cnpy)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rdparty/cnpy)

link_directories(${CMAKE_SOURCE_DIR}/lib)

set(LINK_LIBS rknnrt)

add_executable(rsvp_rknn_async
    src/main.cpp
    src/data_preprocessor.cpp
    src/inference_engine.cpp
    src/performance_evaluator.cpp
    src/async_pipeline.cpp
    src/utils.cpp
)

target_link_libraries(rsvp_rknn_async ${LINK_LIBS} cnpy pthread stdc++fs)

