set(
  SOURCES

  SolveEquations.cpp
)

set(
  HEADERS

  Constants.h

  SolveEquations.h
  
  Vector.h
  impl/VectorImpl.h
  impl/Vector2dImpl.h
  impl/Vector3dImpl.h
  VectorOperations.h
  impl/VectorOperationsImpl.h
  
  SquareMatrix.h
  impl/SquareMatrixImpl.h
)

set(
  PY
  
  math_py.hpp
  Vector_py.hpp
)

set(
  TESTS

  Vector3d_test.cpp
  
  Matrix_test.cpp
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  TESTS   ${TESTS}
)