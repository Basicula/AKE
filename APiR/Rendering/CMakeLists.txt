set(
  SOURCES
  
  Camera.cpp
  
  RenderableObject.cpp
  
  KDTree.cpp
  
  IRenderer.cpp
  CPURenderer.cpp
  CUDARenderer.cpp
  
  Scene.cpp
)

set(
  HEADERS
  
  Camera.h
  IRenderable.h
  RenderableObject.h
  
  KDTree.h
  
  IRenderer.h
  CPURenderer.h
  CUDARenderer.h
  
  Scene.h
)

set(
  PY
  
  Camera_py.hpp
  IRenderable_py.hpp
  RenderableObject_py.hpp
  Scene_py.hpp
  rendering_py.hpp
)

generate_project(
  STATIC
  SUPPORT_CUDA
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  LINK    Common
          Geometry
          Image
          Math
          Memory
          Visual
)