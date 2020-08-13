set(
  SOURCES
  
  Simulation.cpp
)

set(
  HEADERS
  
  Constants.h
  
  Simulation.h
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    Math
)