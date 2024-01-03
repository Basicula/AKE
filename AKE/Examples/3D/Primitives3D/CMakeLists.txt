set(
  SOURCES
  
  main.cpp
  PrimitivesRendering.cpp
  SimpleCameraController.cpp
)

set(
  HEADERS
  
  PrimitivesRendering.h
  SimpleCameraController.h
)

set(
  LINKS
  
  Common
  Image
  Math
  Memory
  Physics
  Geometry
  Geometry.3D
  Rendering
  Visual
  Window
)

set(NAME Primitives3D)

add_executable(
  ${NAME}
  ${SOURCES}
  ${HEADERS}
)

target_link_libraries(${NAME} PRIVATE ${LINKS})

set_property(TARGET ${NAME} PROPERTY FOLDER "AKE/Examples/3D")

add_custom_command(TARGET ${NAME} POST_BUILD
                   COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${GLEW_BINARY}" "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/"
)
