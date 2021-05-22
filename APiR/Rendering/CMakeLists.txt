set(
  SOURCES
  
  Camera.cpp
  
  RenderableObject.cpp
  
  Container.cpp
  KDTree.cpp
  
  IRenderer.cpp
  CPURayTracer.cpp
  
  Scene.cpp
)

set(
  HEADERS
  
  Camera.h
  IRenderable.h
  RenderableObject.h
  
  Container.h
  KDTree.h
  
  IRenderer.h
  CPURayTracer.h
  
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

set(
  CUDA_FILES
)

set(
  LINKS
  
  Common
  Geometry
  Image
  Math
  Memory
  Visual
)

if(ENABLE_CUDA)
list(APPEND HEADERS CUDARayTracer.h)
list(APPEND SOURCES CUDARayTracer.cpp)
list(APPEND CUDA_FILES src/CUDARayTracer.cpp)
endif()

generate_project(
  STATIC
  SUPPORT_CUDA
  CUDA_FILES  ${CUDA_FILES}
  SOURCES     ${SOURCES}
  HEADERS     ${HEADERS}
  PY          ${PY}
  LINK        ${LINKS}
)