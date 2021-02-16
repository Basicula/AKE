set(
  SOURCES

  Color.cpp
  
  SpotLight.cpp
  
  ColorMaterial.cpp  
)

set(
  HEADERS
  
  Color.h
  
  ILight.h
  SpotLight.h
  
  IMaterial.h
  ColorMaterial.h
)

set(
  PY
  
  Color_py.hpp
  
  material_py.hpp
  IMaterial_py.hpp
  ColorMaterial_py.hpp
  
  light_py.hpp
  ILight_py.hpp  
  SpotLight_py.hpp
  
  visual_py.hpp
)

generate_project(
  STATIC
  SUPPORT_CUDA
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  LINK    Math
)