set(
  SOURCES

  Collider2D.cpp
  CircleCollider.cpp

  Collision2D.cpp
  CollisionDetection2D.cpp
  GJKCollisionDetection2D.cpp
  GJKCircleConvex2D.cpp

  PhysicalProperty2D.cpp
)

set(
  HEADERS

  Collider2D.h
  CircleCollider.h

  Collision2D.h
  CollisionDetection2D.h
  GJKCollisionDetection2D.h
  GJKConvex2D.h
  GJKCircleConvex2D.h

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
