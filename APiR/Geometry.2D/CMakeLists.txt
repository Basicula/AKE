set(
  SOURCES

  Shape.cpp
  Circle.cpp
  Rectangle.cpp
)

set(
  HEADERS

  Shape.h
  Circle.h
  Rectangle.h
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
