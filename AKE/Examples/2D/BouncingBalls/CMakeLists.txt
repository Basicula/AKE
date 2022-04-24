set(
  SOURCES
  
  main.cpp

  World.cpp
  WorldController.cpp
  WorldRenderer.cpp
  GUI.cpp
)

set(
  HEADERS
  
  World.h
  WorldController.h
  WorldRenderer.h
  GUI.h
)

set(
  LINKS

  Physics
  Physics.2D
  Geometry
  Geometry.2D
  Window
  Math
  Renderer.2D
  Rendering
  Rendering.2D
  Common
  Visual
  World.2D
)

set(NAME BouncingBalls)

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
