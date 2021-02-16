set(
  SOURCES
  
  main.cpp
  
  CudaTest.h
  CudaTest.cu
)

if (WIN32)
generate_project(
  EXECUTABLE
  SUPPORT_CUDA
  SOURCES ${SOURCES}
  LINK    BMPWriter
          CUDACore
          Fluid
            Physics
          Fractal
          Geometry
          GLUTWindow
          Math
          OpenCLCore
          Rendering
            Common
          Scene
          Visual
)
file(COPY ${GLUT_BINARY} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()