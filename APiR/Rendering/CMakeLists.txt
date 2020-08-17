set(
  SOURCES
  
  Camera.cpp
  Image.cpp
  KDNode.cpp
  KDTree.cpp
  RenderableObject.cpp
)

set(
  HEADERS
  
  Camera.h
  Image.h
  IRenderable.h
  KDNode.h
  KDTree.h
  RenderableObject.h
)

set(
  PY
  
  Camera_py.hpp
  Image_py.hpp
  IRenderable_py.hpp
  RenderableObject_py.hpp
  rendering_py.hpp
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  LINK    Common
          Geometry
          Math
          Visual  
)