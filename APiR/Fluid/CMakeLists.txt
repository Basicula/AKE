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

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  LINK    Common
          Geometry
          Math
          Physics
          Rendering
            Visual
)