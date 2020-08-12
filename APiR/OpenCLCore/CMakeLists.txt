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

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    Math
          Visual
          ${OPENCL_LIBRARIES}
)