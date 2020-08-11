set(
  SOURCES
  
  BMPWriter.cpp
)

set(
  HEADERS
  
  BMPWriter.h
)

set(
  PY
)

set(
  TESTS
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  TESTS   ${TESTS}
  LINK    Math
          Visual
)