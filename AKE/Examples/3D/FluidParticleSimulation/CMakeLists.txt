set(
  SOURCES
  
  main.cpp
  FluidExample.cpp
)

set(
  HEADERS
  
  FluidExample.h
)

set(
  LINKS
  
  Common
  Image
  Math
  Memory
  Fluid
  Physics
  Geometry
  Geometry.3D
  Rendering
  Visual
  Window
)

set(NAME FluidParticleSimulation)

add_executable(
  ${NAME}
  ${SOURCES}
  ${HEADERS}
)

target_link_libraries(${NAME} PRIVATE ${LINKS})

set_property(TARGET ${NAME} PROPERTY FOLDER "AKE/Examples/3D")

add_custom_command(TARGET ${NAME} POST_BUILD
                   COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${GLEW_BINARY}" "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/"
                   COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${GLUT_BINARY}" "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/"
)
