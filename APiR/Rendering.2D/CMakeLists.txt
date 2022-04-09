set(
  SOURCES

  Drawer.cpp
  CircleDrawer.cpp
  RectangleDrawer.cpp
  Triangle2DDrawer.cpp

  OpenGLRenderer.cpp

  Object2D.cpp

  Scene2D.cpp
)

set(
  HEADERS
  
  Drawer.h
  CircleDrawer.h
  RectangleDrawer.h
  Triangle2DDrawer.h

  OpenGLRenderer.h

  Object2D.h
  impl/Object2DImpl.h

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
