set(
  SOURCES

  main.cpp
  
  SceneController.cpp
  FillScene.cpp
)

set(
  HEADERS

  SceneController.h
  FillScene.h
)

set(
  LINKS

  Common
  Math
  Geometry
  Geometry.2D
  Renderer.2D
  Rendering
  Rendering.2D
  Physics.2D
  Visual
  Window
  World.2D
  Utils2D
)

set(NAME GJKCollisionDetection)

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

