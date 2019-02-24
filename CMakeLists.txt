cmake_minimum_required(VERSION 3.12)

project(RayTracing)

add_subdirectory(BMPWriter)

add_executable(RayTracing main.cpp)

target_link_libraries(RayTracing PRIVATE BMPWriter)