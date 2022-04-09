set(
  SOURCES
  
  Circle.cpp
  Rectangle.cpp
  Triangle2D.cpp
)

set(
  HEADERS

  Shape2D.h
  Circle.h
  Rectangle.h
  Triangle2D.h
)

set(
  LINKS

  Common
  Geometry
  Math
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    ${LINKS}
)
