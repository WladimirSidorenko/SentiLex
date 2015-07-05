cmake_minimum_required(VERSION 2.8.8)

project(vec2dic)

add_definitions(-Wall -std=c++11)

set(EXECUTABLE_OUTPUT_PATH bin)
set(SOURCE src/vec2dic.cpp)

add_executable(vec2dic ${SOURCE})
