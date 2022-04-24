set(
  SOURCES
  
  src/Utils2D.cpp
)

set(
  HEADERS

  include/Utils2D/Utils2D.h
)

set(
  LINKS

  Common
  Math
  Geometry
  Geometry.2D
  Rendering
  Rendering.2D
  Physics.2D
  Visual
  Window
  World.2D
)

set(NAME Utils2D)

add_library(
  ${NAME}
  STATIC
  ${SOURCES}
  ${HEADERS}
)

target_link_libraries(${NAME} PRIVATE ${LINKS})
target_include_directories(${NAME} PUBLIC "include")

set_property(TARGET ${NAME} PROPERTY FOLDER "AKE/Examples/2D")
