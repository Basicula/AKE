set(
  SOURCES
  
  MandelbrotSetKernel.cpp
)

set(
  HEADERS
  
  MandelbrotSetKernel.h
)

add_definitions(-DKERNELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/Kernels")
add_definitions(-DMANDELBROT_SET_KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/Kernels/MandelbrotSetKernel.cl")

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  TESTS   ${TESTS}
  LINK    Image
          Math
          Memory
          OpenCLCore
          Visual
)