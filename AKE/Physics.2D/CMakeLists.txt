set(
  SOURCES

  Collider2D.cpp
  
  GJKCollisionDetection2D.cpp
  GJKCircleConvex2D.cpp
  GJKRectangleConvex2D.cpp
  GJKTriangleConvex2D.cpp
)

set(
  HEADERS

  Collider2D.h

  Collision2D.h
  GJKCollisionDetection2D.h
  GJKConvex2D.h
  GJKCircleConvex2D.h
  GJKRectangleConvex2D.h
  GJKTriangleConvex2D.h

  PhysicalProperty2D.h
)

set(
  LINKS

  Geometry
  Geometry.2D
  Math
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    ${LINKS}
)
