set(
  SOURCES
  
  ThreadPool.cpp
)

set(
  HEADERS
  
  ThreadPool.h
  impl/ThreadPool_impl.h
  ParallelUtils.h
  
  IObject.h
  
  Constants.h
)

set(
  PY
  
  common_py.hpp
  
  IObject_py.hpp
)


set(
  TESTS
  
  ThreadPool_test.cpp
  
  Utils.h
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  TESTS   ${TESTS}
)

