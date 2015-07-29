cmake_minimum_required(VERSION 2.8.8)

project(vec2dic)

find_package(Armadillo REQUIRED)
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
	add_custom_target(doc
		${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generating API documentation with Doxygen" VERBATIM
		)
endif(DOXYGEN_FOUND)

include_directories(src/)
set(EXECUTABLE_OUTPUT_PATH bin)

add_definitions(-Wall -Wextra -std=c++11 -fopenmp -O1)

set(SOURCES src/vec2dic.cpp src/expansion.h src/expansion.cpp)
add_executable(vec2dic ${SOURCES})
target_link_libraries(vec2dic ${ARMADILLO_LIBRARIES})
