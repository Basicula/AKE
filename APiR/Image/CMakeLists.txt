set(
  SOURCES
  
  Image.cpp
)

set(
  HEADERS
  
  Image.h
)

set(
  PY
  
  Image_py.hpp
  image_module_py.hpp
)

set(
  CUDA_FILES
  
  src/Image.cpp
)

generate_project(
  STATIC
  SUPPORT_CUDA
  CUDA_FILES ${CUDA_FILES}
  SOURCES    ${SOURCES}
  HEADERS    ${HEADERS}
  PY         ${PY}
  LINK       Math
             Memory
             Visual
)