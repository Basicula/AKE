set(
  SOURCES

  Window.cpp
  GLFWWindow.cpp
  GLUTWindow.cpp

  GLFWGUIView.cpp
  GLFWDebugGUIView.cpp

  EventListner.cpp
  Event.cpp
  KeyboardEvent.cpp
  MouseEvent.cpp
  WindowEvent.cpp

  FPSCounter.cpp
)

set(
  HEADERS

  Window.h
  impl/WindowImpl.h
  GLFWWindow.h
  GLUTWindow.h

  GUIView.h
  GLFWGUIView.h
  GLFWDebugGUIView.h

  EventListner.h
  Event.h
  KeyboardEvent.h
  MouseEvent.h
  WindowEvent.h

  Keys.h

  FPSCounter.h
)

set(
  LINKS
  
  Image
  Math
  Memory
  Rendering
  Visual
)

set(
  PUBLIC_LINKS
  
  glfw
  ${GLUT_LIBRARIES}
  imgui
)

generate_project(
  STATIC
  SOURCES     ${SOURCES}
  HEADERS     ${HEADERS}
  LINK        ${LINKS}
  PUBLIC_LINK ${PUBLIC_LINKS}
)