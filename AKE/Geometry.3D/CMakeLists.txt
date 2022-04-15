set(
  SOURCES

  Transformable.cpp
  
  ISurface.cpp
  Sphere.cpp
  Torus.cpp
  Cylinder.cpp
  Plane.cpp
  Ray.cpp
  
  Intersection.cpp
)

set(
  HEADERS

  Transformable.h
  
  Cylinder.h
  ISurface.h
  Plane.h
  Ray.h
  Sphere.h
  Torus.h
  
  Intersection.h
)

set(
  PY
  
  Intersection_py.hpp
  ISurface_py.hpp
  Ray_py.hpp
  Sphere_py.hpp
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
