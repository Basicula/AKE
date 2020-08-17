set(
  SOURCES
  
  BoundingBox.cpp
  Cylinder.cpp
  Intersection.cpp
  Plane.cpp
  Ray.cpp
  Sphere.cpp
  Torus.cpp
  Transformable.cpp
  Transformation.cpp
)

set(
  HEADERS
  
  BoundingBox.h
  Cylinder.h
  Intersection.h
  ISurface.h
  Plane.h
  Ray.h
  Sphere.h
  Torus.h
  Transformable.h
  Transformation.h
)

set(
  PY
  
  BoundingBox_py.hpp
  Intersection_py.hpp
  ISurface_py.hpp
  geometry_py.hpp
  Ray_py.hpp
  Sphere_py.hpp
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  LINK    Math
          Common
)