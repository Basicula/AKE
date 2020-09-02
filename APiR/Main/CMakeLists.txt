set(
  SOURCES
  
  main.cpp
)

if (WIN32)
generate_project(
  EXECUTABLE
  SOURCES ${SOURCES}
  LINK    BMPWriter
          Fluid
            Physics
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