set(
  SOURCES

  Circle.cpp
  Shape.cpp
)

set(
  HEADERS

  Circle.h
  Shape.h
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
