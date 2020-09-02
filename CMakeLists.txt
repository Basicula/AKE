cmake_minimum_required(VERSION 3.8)

project(APiR)

set(CMAKE_CXX_STANDARD 17)
set(OpenGL_GL_PREFERENCE LEGACY)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -pthread -lGL -lGLU -lglut")
endif()

set(THIRD_PARTIES "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties" CACHE PATH "" FORCE)
set(CMAKE_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE PATH "" FORCE)

# find all needed packages
find_package(OpenGL REQUIRED)
include("${CMAKE_FOLDER}/glut.cmake")
include("${CMAKE_FOLDER}/OpenCL.cmake")

# for python integration
add_subdirectory(pybind11)

# aditional packages
add_subdirectory(cmake)

# to future testing libs
enable_testing()

# main lib
include("cmake/generate_project.cmake")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
add_subdirectory(APiR)
