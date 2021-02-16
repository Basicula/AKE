set(
  SOURCES
  
	GLUTWindow.cpp
	FPSCounter.cpp
)

set(
  HEADERS
  
	GLUTWindow.h
	FPSCounter.h
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    CUDACore
          Rendering
            Math
            Visual
          ${GLUT_LIBRARIES}
)