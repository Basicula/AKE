set(
  SOURCES
  
  ColorMap.cpp
  DefaultColorMap.cpp
  SmoothColorMap.cpp
  
  Fractal.cu
  MandelbrotSet.cu
  JuliaSet.cpp
  LyapunovFractal.cpp
)

set(
  HEADERS
  
  ColorMap.h
  DefaultColorMap.h
  SmoothColorMap.h
  
  Fractal.h
  MandelbrotSet.h
  JuliaSet.h
  LyapunovFractal.h
)

generate_project(
  STATIC
  SUPPORT_CUDA
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    CUDACore
          Visual
            Math
)