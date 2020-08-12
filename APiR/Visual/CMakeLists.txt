set(
  SOURCES

  Camera.cpp
  Color.cpp
  ColorMaterial.cpp
  Image.cpp
  SpotLight.cpp
  
  RenderableObject.cpp
  
  KDTree.cpp
  KDNode.cpp
)

set(
  HEADERS
  
  ILight.h
  IMaterial.h
  
  IRenderable.h
  RenderableObject.h
  
  KDTree.h
  KDNode.h

  Camera.h
  Color.h
  ColorMaterial.h
  Image.h
  SpotLight.h
)

set(
  PY
  
  Camera_py.hpp
  ColorMaterial_py.hpp
  Color_py.hpp
  ILight_py.hpp
  Image_py.hpp
  IMaterial_py.hpp
  IRenderable_py.hpp
  light_py.hpp
  material_py.hpp
  RenderableObject_py.hpp
  renderable_py.hpp
  SpotLight_py.hpp
  visual_py.hpp
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  LINK    Math
          Common
          Primitives
)