cmake_minimum_required(VERSION 3.10)
project(Convolution)

set(CMAKE_CXX_STANDARD 11)

add_executable(Convolution temp.cpp main.cpp convolution.h)

find_package(OpenCL REQUIRED)
target_link_libraries(Convolution OpenCL::OpenCL)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Convolution ${OpenCV_LIBS})