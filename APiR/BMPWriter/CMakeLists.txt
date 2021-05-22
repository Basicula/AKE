set(
  SOURCES
  
  BMPWriter.cpp
)

set(
  HEADERS
  
  BMPWriter.h
)

set(
  LINKS
  
  Image
  Math
  Memory
  Visual
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    ${LINKS}
)