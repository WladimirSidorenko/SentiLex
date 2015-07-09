cmake_minimum_required(VERSION 2.8.8)

project(vec2dic)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH bin)
set(SOURCES src/vec2dic.cpp src/expansion.cpp)

add_definitions(-Wall -std=c++11)
add_executable(vec2dic ${SOURCES})
