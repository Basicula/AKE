set(
  SOURCES
  
  BMPWriter.cpp
)

set(
  HEADERS
  
  BMPWriter.h
)

generate_project(
  STATIC
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  LINK    Image
          Math
          Memory
          Visual
)