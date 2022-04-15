set(
  SOURCES
  
  Simulation.cpp
)

set(
  HEADERS
  
  Constants.h
  
  Simulation.h
  
  IPhysicMaterial.h
)

set(
  LINKS
  
  Math
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    ${LINKS}
)