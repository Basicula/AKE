set(
  SOURCES
  
  ColorMap.cpp
  DefaultColorMap.cpp
  SmoothColorMap.cpp
  
  Fractal.cpp
  MandelbrotSet.cpp
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
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    Visual
            Math
)