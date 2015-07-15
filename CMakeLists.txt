cmake_minimum_required(VERSION 2.8.8)

project(vec2dic)

set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} armadillo)
set(EXECUTABLE_OUTPUT_PATH bin)
set(SOURCE src/vec2dic.cpp)

add_definitions(-Wall -std=c++11)
add_executable(vec2dic ${SOURCE})
