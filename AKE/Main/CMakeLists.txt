set(
  SOURCES
  
  main.cpp
  
  SceneExamples.cpp
  
  ConsoleLogEventListner.cpp
  SimpleCameraController.cpp

  FluidExample.cpp
  2DSceneExample.cpp
  CollisionDetectionExample2D.cpp
  3DSceneExample.cpp
  OpenCLExample.cpp
  FractalExample.cpp
  WindowExample.cpp

  Utils2D.h
  Utils2D.cpp
)

set(
  HEADERS
  
  SceneExamples.h
  
  ConsoleLogEventListner.h
  SimpleCameraController.h

  FluidExample.h
  Example2D.h
  3DSceneExample.h
  OpenCLExample.h
  FractalExample.h
  WindowExample.h
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
  Physics.2D
  Fractal
  Geometry
  Geometry.2D
  Geometry.3D
  Window
  Image
  Math
  Memory
  Renderer.2D
  Rendering
  Rendering.2D
  Common
  Visual
  World.2D
)

if(ENABLE_CUDA)
list(APPEND SOURCES CudaTest.cpp)
list(APPEND HEADERS CudaTest.h)
endif()

if(ENABLE_OPENCL)
  list(APPEND LINKS OpenCLCore
                    OpenCLKernels)
endif()

if (WIN32)
generate_project(
  EXECUTABLE
  SUPPORT_CUDA
  CUDA_FILES    ${CUDA_FILES}
  SOURCES       ${SOURCES}
  HEADERS       ${HEADERS}
  LINK          ${LINKS}
)

add_custom_command(TARGET Main POST_BUILD
                   COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${GLUT_BINARY}" "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/"
                   COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${GLEW_BINARY}" "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/"
)

endif()