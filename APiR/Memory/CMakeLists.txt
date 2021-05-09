set(
  SOURCES
  
  MemoryManager.cpp
)

set(
  HEADERS
  
  custom_vector.h
  impl/custom_vector_impl.h
  
  default_allocator.h
  impl/default_allocator_impl.h
  device_allocator.h
  impl/device_allocator_impl.h
  managed_allocator.h
  impl/managed_allocator_impl.h
  
  base_ptr.h
  impl/base_ptr_impl.h
  device_ptr.h
  impl/device_ptr_impl.h
  managed_ptr.h
  impl/managed_ptr_impl.h
  
  MemoryManager.h
  impl/MemoryManager_impl
)

set(
  TESTS
  
  custom_vector_test.cpp
  
  TestKernels.h
  TestKernels.cu

  device_ptr_test.cpp
  managed_ptr_test.cpp
)

set(
  CUDA_FILES
  
  src/MemoryManager.cpp
  
  tests/c++/device_ptr_test.cpp
  tests/c++/managed_ptr_test.cpp
)

generate_project(
  STATIC
  SUPPORT_CUDA
  CUDA_FILES    ${CUDA_FILES}
  SOURCES       ${SOURCES}
  HEADERS       ${HEADERS}
  TESTS         ${TESTS}
  PUBLIC_LINK   CUDACore
)