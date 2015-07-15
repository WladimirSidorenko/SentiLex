cmake_minimum_required(VERSION 2.8.8)

project(vec2dic)

find_package(Armadillo REQUIRED)

include_directories(src/)
set(EXECUTABLE_OUTPUT_PATH bin)
set(SOURCES src/vec2dic.cpp src/expansion.h src/expansion.cpp)

add_definitions(-Wall -Wextra -std=c++11 -fopenmp)
add_executable(vec2dic ${SOURCES})
