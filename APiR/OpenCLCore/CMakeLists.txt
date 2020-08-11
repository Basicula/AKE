set(
  SOURCES
  
  src/OpenCLEnvironment.cpp
  
  src/KernelSize.cpp
  src/Kernel.cpp
  src/MandelbrotSetKernel.cpp
  
  src/OpenCLUtils.cpp
)

set(
  HEADERS

  include/OpenCLEnvironment.h

  include/KernelSize.h
  include/Kernel.h
  include/MandelbrotSetKernel.h
  
  include/OpenCLUtils.h
)

add_library(
  OpenCLCore

  ${SOURCES}
  ${HEADERS}
)

target_include_directories(OpenCLCore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(OpenCLCore PRIVATE ${OPENCL_LIBRARIES})
target_link_libraries(OpenCLCore PRIVATE Math)
target_link_libraries(OpenCLCore PRIVATE Visual)