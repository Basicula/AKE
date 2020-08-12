set(
  SOURCES
    
	Sphere.cpp
	Plane.cpp
	Torus.cpp
	Cylinder.cpp
)

set(
  HEADERS
  
  ISurface.h
  
	Sphere.h
	Plane.h
	Torus.h
	Cylinder.h
)

set(
  PY
  
  ISurface_py.hpp
  primitives_py.hpp
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