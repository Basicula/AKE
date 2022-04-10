set(
  SOURCES

  OpenGLRenderer.cpp
)

set(
  HEADERS

  OpenGLRenderer.h
)

set(
  LINKS

  Geometry
  Geometry.2D
  Rendering
  Rendering.2D
  World.2D
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    ${LINKS}
)
