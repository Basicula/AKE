set(
  SOURCES
  
  Camera.cpp
  Image.cpp
  KDTree.cpp
  RenderableObject.cpp
)

set(
  HEADERS
  
  Camera.h
  Image.h
  IRenderable.h
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
  SUPPORT_CUDA
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  LINK    Common
          CUDACore
          Geometry
          Math
          Visual
)