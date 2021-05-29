set(
  SOURCES

  Color.cpp
  
  SpotLight.cpp
  
  PhongMaterial.cpp  
)

set(
  HEADERS
  
  Color.h
  
  ILight.h
  SpotLight.h
  
  IVisualMaterial.h
  PhongMaterial.h
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

set(
  CUDA_FILES
  
  src/Color.cpp
)

set(
  LINKS
  
  Math
  Common
)

generate_project(
  STATIC
  SUPPORT_CUDA
  CUDA_FILES    ${CUDA_FILES}
  SOURCES       ${SOURCES}
  HEADERS       ${HEADERS}
  PY            ${PY}
  LINK          ${LINKS}
)