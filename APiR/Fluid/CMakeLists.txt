set(
  SOURCES
  
  Fluid.cpp
  
  BFPointSearcher.cpp
  
  ParticleSystem.cpp
  SPHSystem.cpp
  
  SPHSimulation.cpp
  
  SPHKernel.cpp
  SPHStandartKernel.cpp
  SPHSpikyKernel.cpp
)

set(
  HEADERS
  
  Fluid.h
  
  PointNeighborSearcher.h
  BFPointSearcher.h
  
  ParticleSystem.h
  SPHSystem.h
  
  SPHSimulation.h
  
  SPHKernel.h
  SPHStandartKernel.h
  SPHSpikyKernel.h
  
  FluidConstants.h
)

set(
  PY
  
  fluid_module_py.hpp
  Fluid_py.hpp
)

set(
  LINKS

  Common
  Geometry
  Geometry.3D
  Math
  Physics
  Rendering
  Visual
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  LINK    ${LINKS}
)