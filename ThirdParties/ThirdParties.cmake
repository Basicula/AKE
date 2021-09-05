find_package(OpenGL REQUIRED)

# NOTE: need to add ${THIRD_PARTIES} prefix for all pathes due to CMake include behavior

include("${THIRD_PARTIES}/glut.cmake")
include("${THIRD_PARTIES}/glfw.cmake")
include("${THIRD_PARTIES}/glew.cmake")
include("${THIRD_PARTIES}/gtest.cmake")
include("${THIRD_PARTIES}/imgui.cmake")
if (ENABLE_OPENCL)
  add_definitions(-DENABLED_OPENCL)
  include("${THIRD_PARTIES}/OpenCL.cmake")
endif()
if (ENABLE_CUDA)
  add_definitions(-DENABLED_CUDA)
  include("${THIRD_PARTIES}/CUDA.cmake")
endif()

if (ENABLE_PYTHON_WRAPPER)
# for python integration
add_subdirectory(${THIRD_PARTIES}/pybind11)
endif()