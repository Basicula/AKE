set(
  SOURCES
  
  Scene.cpp
)

set(
  HEADERS
  
  Scene.h
)

set(
  PY
  
  scene_module_py.hpp
  Scene_py.hpp
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  LINK    CUDACore
          Rendering
            Common
            Geometry
          Visual
            Math
)