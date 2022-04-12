set(
  SOURCES

  Object2D.cpp

  Scene2D.cpp
)

set(
  HEADERS

  Object2D.h
  impl/Object2DImpl.h

  Scene2D.h
)

set(
  LINKS

  Geometry
  Geometry.2D
  Rendering.2D
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    ${LINKS}
)
