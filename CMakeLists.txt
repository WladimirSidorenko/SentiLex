cmake_minimum_required(VERSION 2.8.8)

project(vec2dic)

find_package(Armadillo REQUIRED)
find_package(OpenMP)
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
	add_custom_target(doc
		${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generating API documentation with Doxygen" VERBATIM
		)
endif(DOXYGEN_FOUND)

add_definitions(-Wall -Wextra -std=c++11 -fopenmp  -funroll-loops #
		      -march=native -funroll-loops -Ofast)

# set variables
set(V2D_SRC_DIR src/vec2dic CACHE FILEPATH "Default directory containing vec2dic source files")
set(W2V_SRC_DIR src/word2vec CACHE FILEPATH "Default directory containing word2vec source files")

# specify target directory
set(EXECUTABLE_OUTPUT_PATH bin)

# define targets
aux_source_directory(${V2D_SRC_DIR} V2D_SOURCES)
add_executable(vec2dic ${V2D_SOURCES})
target_include_directories(vec2dic PRIVATE ${V2D_SRC_DIR})
target_link_libraries(vec2dic ${ARMADILLO_LIBRARIES})

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
aux_source_directory(${W2V_SRC_DIR} W2V_SOURCES)
add_executable(word2vec ${W2V_SOURCES})
include_directories(word2vec PRIVATE ${W2V_SRC_DIR} #
			     PRIVATE ${W2V_SRC_DIR}/flatbuffers #
			     PRIVATE ${W2V_SRC_DIR}/utf8cpp #
				   PRIVATE ${W2V_SRC_DIR}/utf8cpp/utf8)
