set(
  SOURCES
  
  MappingFunctions.cpp
  
  Fractal.cpp
  MandelbrotSet.cpp
  JuliaSet.cpp
  LyapunovFractal.cpp
)

set(
  HEADERS
  
  MappingFunctions.h
  
  Fractal.h
  MandelbrotSet.h
  JuliaSet.h
  LyapunovFractal.h
)

set(
  CUDA_FILES
  
  src/MappingFunctions.cpp
  
  src/Fractal.cpp
  src/MandelbrotSet.cpp
  src/JuliaSet.cpp
  src/LyapunovFractal.cpp
)

set(
  LINKS
  
  Common
  Memory
  Visual
  Math
)

generate_project(
  STATIC
  SUPPORT_CUDA
  CUDA_FILES    ${CUDA_FILES}
  SOURCES       ${SOURCES}
  HEADERS       ${HEADERS}
  LINK          ${LINKS}
)