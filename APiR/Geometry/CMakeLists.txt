set(
  SOURCES
  
)

set(
  HEADERS
  
  BoundingBox.h
  impl/BoundingBoxImpl.h
  
  Transformation.h
  impl/TransformationImpl.h
  impl/Transformation2DImpl.h
  impl/Transformation3DImpl.h
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
  INTERFACE
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  LINK    ${LINKS}
)