set(
  SOURCES

  Drawer.cpp
  SimpleDrawer.cpp
  CircleDrawer.cpp
  RectangleDrawer.cpp
  Triangle2DDrawer.cpp
)

set(
  HEADERS
  
  Drawer.h
  SimpleDrawer.h
  CircleDrawer.h
  RectangleDrawer.h
  Triangle2DDrawer.h
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
