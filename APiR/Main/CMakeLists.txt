set(
  SOURCES
  
  main.cpp
  
  SceneExamples.cpp
)

set(
  HEADERS
  
  SceneExamples.h
)

set(
  CUDA_FILES
  
  src/CudaTest.cpp
)

set(
  LINKS

  BMPWriter
  Fluid
  Physics
  Fractal
  Geometry
  GLUTWindow
  Image
  Math
  Memory
  Rendering
  Common
  Visual
)

if(ENABLE_CUDA)
list(APPEND SOURCES CudaTest.cpp)
list(APPEND HEADERS CudaTest.h)
endif()

if(ENABLED_OPENCL)
  list(APPEND LINKS OpenCLCore
                    OpenCLKernels)
endif()

if (WIN32)
generate_project(
  EXECUTABLE
  SUPPORT_CUDA
  CUDA_FILES ${CUDA_FILES}
  SOURCES    ${SOURCES}
  HEADERS    ${HEADERS}
  LINK       ${LINKS}
)
file(COPY ${GLUT_BINARY} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()