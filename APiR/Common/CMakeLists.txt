set(
  SOURCES
  
  BoundingBox.cpp
  Intersection.cpp
  Ray.cpp
  
  BFPointSearcher.cpp
  
  Simulation.cpp
  
  ThreadPool.cpp
  
  Transformable.cpp
  Transformation.cpp
)

set(
  HEADERS
    
  BoundingBox.h
  Intersection.h
  Ray.h

  PointNeighborSearcher.h
  BFPointSearcher.h
  
  Simulation.h
  
  ThreadPool.h
  impl/ThreadPool_impl.h
  ParallelUtils.h
  
  Transformable.h
  Transformation.h
  
  IObject.h
  
  DefinesAndConstants.h
)

set(
  PY
  
  common_py.hpp
  
  BoundingBox_py.hpp
  Intersection_py.hpp
  IObject_py.hpp
  Ray_py.hpp
)


set(
  TESTS
  
  ThreadPool_test.cpp
  Transformation_test.cpp
  
  Utils.h
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  TESTS   ${TESTS}
  LINK    Math
)

