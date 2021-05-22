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

set(
  LINKS
  
  Image
  Math
  Memory
  Visual
  ${GLUT_LIBRARIES}
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    ${LINKS}
)