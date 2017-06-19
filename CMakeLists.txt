project(FACE)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include(mini-caffe/mini-caffe.cmake)

set(BOOST_DIR $ENV{BOOST_DIR})
SET(CMAKE_LIBRARY_PATH ${BOOST_DIR}/stage/lib)
include_directories(${BOOST_DIR})

file(GLOB HEADERS *.h*)
file(GLOB SOURCES *.c*)
set(ALL_FILES ${HEADERS} ${SOURCES})

add_executable(${PROJECT_NAME} ${ALL_FILES})

target_link_libraries(${PROJECT_NAME} libcaffe ${OpenCV_LIBS} ${Boost_LIBS})
