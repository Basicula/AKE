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
  
  base_ptr.h
  impl/base_ptr_impl.h
  
  MemoryManager.h
  impl/MemoryManager_impl.h
)

set(
  TESTS
  
  Utils.h
  
  custom_vector_test.cpp
)

set(
  CUDA_FILES
  
  src/MemoryManager.cpp
  
  tests/c++/device_ptr_test.cpp
  tests/c++/managed_ptr_test.cpp
)

set(
  PUBLIC_LINKS
)

if(ENABLE_CUDA)
list(APPEND HEADERS device_allocator.h
                    impl/device_allocator_impl.h
                    managed_allocator.h
                    impl/managed_allocator_impl.h
                    device_ptr.h
                    impl/device_ptr_impl.h
                    managed_ptr.h
                    impl/managed_ptr_impl.h
)

list(APPEND TESTS TestKernels.h
                  TestKernels.cu
                  device_ptr_test.cpp
                  managed_ptr_test.cpp
                  custom_vector_cuda_test.cpp
)

list(APPEND PUBLIC_LINKS CUDACore)
endif()

generate_project(
  STATIC
  SUPPORT_CUDA
  CUDA_FILES    ${CUDA_FILES}
  SOURCES       ${SOURCES}
  HEADERS       ${HEADERS}
  TESTS         ${TESTS}
  PUBLIC_LINK   ${PUBLIC_LINKS}
)