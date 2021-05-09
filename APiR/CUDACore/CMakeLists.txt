set(
  SOURCES
  
  CUDAUtils.cpp
)

set(
  HEADERS
  
  
  CUDAUtils.h
)

generate_project(
  STATIC
  SUPPORT_CUDA
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
)