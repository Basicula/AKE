set(
  SOURCES

  Drawer.cpp
  CircleDrawer.cpp
  RectangleDrawer.cpp
  Triangle2DDrawer.cpp
)

set(
  HEADERS
  
  Drawer.h
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
