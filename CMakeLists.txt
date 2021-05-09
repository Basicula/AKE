cmake_minimum_required(VERSION 3.18)

project(APiR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(OpenGL_GL_PREFERENCE LEGACY)

if(MSVC)
  set(CMAKE_CXX_FLAGS           "${CMAKE_CXX_FLAGS} /W4 /WX")
  set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS} /MT /O2 ${INTRINSICS} /Zi /MP /GL /DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG     "${CMAKE_CXX_FLAGS} /MTd /Od /Ob0 ${INTRINSICS} /RTC1 /Zi /D_DEBUG")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -pthread -lGL -lGLU -lglut")
endif()

set(THIRD_PARTIES "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParties" CACHE PATH "" FORCE)
set(CMAKE_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE PATH "" FORCE)

option(ENABLE_CUDA "Turn on/off CUDA usage")
option(ENABLE_OPENCL "Turn on/off CUDA usage")

# find all needed packages
find_package(OpenGL REQUIRED)

# Macros and other predefined stuff
include_directories(${CMAKE_FOLDER}/include)

include("${CMAKE_FOLDER}/glut.cmake")
if (ENABLE_OPENCL)
  add_definitions(-DENABLED_OPENCL)
  include("${CMAKE_FOLDER}/OpenCL.cmake")
endif()
if (ENABLE_CUDA)
  add_definitions(-DENABLED_CUDA)
  include("${CMAKE_FOLDER}/CUDA.cmake") 
endif()

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
