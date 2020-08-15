cmake_minimum_required(VERSION 3.8)

project(APiR)

set(CMAKE_CXX_STANDARD 17)
set(OpenGL_GL_PREFERENCE LEGACY)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -pthread -lGL -lGLU -lglut")
endif()

set(THIRD_PARTIES "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties")

# for python integration
add_subdirectory(pybind11)

# aditional packages
add_subdirectory(cmake)

include_directories(${OPENCL_INCLUDE_DIRS})
link_directories(${OPENCL_LIBRARY_DIRS})

include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})

# to future testing libs
enable_testing()

# main lib
include("cmake/generate_project.cmake")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
add_subdirectory(APiR)
