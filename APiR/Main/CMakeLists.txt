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
endif()