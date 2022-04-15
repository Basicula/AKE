set(
  SOURCES
  
  OpenCLEnvironment.cpp
  
  KernelSize.cpp
  Kernel.cpp
  
  OpenCLUtils.cpp
)

set(
  HEADERS

  OpenCLEnvironment.h

  KernelSize.h
  Kernel.h
  
  OpenCLUtils.h
)

set(
  LINKS
  
  ${OpenCL_LIBRARIES}
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    ${LINKS}
)