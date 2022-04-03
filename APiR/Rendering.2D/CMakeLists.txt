set(
  SOURCES

  Drawer.cpp
  CircleDrawer.cpp

  OpenGLRenderer.cpp

  Scene2D.cpp
)

set(
  HEADERS
  
  Drawer.h
  CircleDrawer.h

  OpenGLRenderer.h

  Scene2D.h
)

set(
  LINKS

  Common
  Geometry
  Geometry.2D
  Math
  Visual
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    ${LINKS}
)
