set(
  SOURCES
  
  CUDAUtils.cpp
  HostDeviceBuffer.cpp
  #MemoryManager.cu
)

set(
  HEADERS
  
  CUDAUtils.h
  HostDeviceBuffer.h
  impl/HostDeviceBufferImpl.h
  HostDevicePtr.h
  impl/HostDevicePtrImpl.h
  #MemoryManager.h
)

generate_project(
  STATIC
  SUPPORT_CUDA
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    
)