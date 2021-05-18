set(
  SOURCES
  
  CUDAUtils.cpp
)

set(
  HEADERS
  
  CUDAUtils.h
  
  KernelHandler.h
  impl/KernelHandler_impl.h
)

generate_project(
  STATIC
  SUPPORT_CUDA
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
)