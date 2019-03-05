cmake_minimum_required(VERSION 3.12)

project(RayTracing)

add_subdirectory(BMPWriter)
add_subdirectory(Math)
add_subdirectory(Primitives)

add_executable(RayTracing main.cpp)

target_link_libraries(RayTracing PRIVATE BMPWriter)
target_link_libraries(RayTracing PRIVATE Math)
target_link_libraries(RayTracing PUBLIC Primitives)