set(
  SOURCES
  
  Window.cpp
  GLFWWindow.cpp
  GLUTWindow.cpp
  Event.cpp
  FPSCounter.cpp
)

set(
  HEADERS
  
  Window.h
  GLFWWindow.h
  GLUTWindow.h
  
  EventListner.h
  Event.h
  
  Keys.h
  
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

set(
  PUBLIC_LINKS
  
  glfw
)

generate_project(
  STATIC
  SOURCES     ${SOURCES}
  HEADERS     ${HEADERS}
  LINK        ${LINKS}
  PUBLIC_LINK ${PUBLIC_LINKS}
)