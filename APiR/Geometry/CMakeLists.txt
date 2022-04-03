set(
  SOURCES
  
  Transformation.cpp
)

set(
  HEADERS
  
  BoundingBox.h
  
  Transformation.h
)

set(
  PY
  
  BoundingBox_py.hpp
  geometry_py.hpp
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