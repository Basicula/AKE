set(
  SOURCES

  main.cpp

  GameController.cpp
  GameRenderer.cpp
  TowerDefense.cpp
)

set(
  HEADERS

  GameController.h
  GameRenderer.h
  TowerDefense.h
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

set(NAME TowerDefense)

add_executable(
  ${NAME}
  ${SOURCES}
  ${HEADERS}
)

target_link_libraries(${NAME} PRIVATE ${LINKS})

set_property(TARGET ${NAME} PROPERTY FOLDER "AKE/Examples/2D")

add_custom_command(TARGET ${NAME} POST_BUILD
                   COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${GLEW_BINARY}" "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/"
)

