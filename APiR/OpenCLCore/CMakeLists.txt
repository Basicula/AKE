set(
  SOURCES
  
  OpenCLEnvironment.cpp
  
  KernelSize.cpp
  Kernel.cpp
  MandelbrotSetKernel.cpp
  
  OpenCLUtils.cpp
)

set(
  HEADERS

  OpenCLEnvironment.h

  KernelSize.h
  Kernel.h
  MandelbrotSetKernel.h
  
  OpenCLUtils.h
)

add_definitions(-DKERNELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/Kernels")
add_definitions(-DMANDELBROT_SET_KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/Kernels/MandelbrotSetKernel.cl")

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    Rendering
            Math
            Visual
          ${OpenCL_LIBRARIES}
)