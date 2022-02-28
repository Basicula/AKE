set(
  SOURCES
  
  Transformable.cpp
  Transformation.cpp
  
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
  
  BoundingBox.h
  
  Transformable.h
  Transformation.h
  
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
  
  BoundingBox_py.hpp
  Intersection_py.hpp
  ISurface_py.hpp
  geometry_py.hpp
  Ray_py.hpp
  Sphere_py.hpp
)

set(
  LINKS
  
  Math
  Common
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  LINK    ${LINKS}
)