set(
  SOURCES
  
  src/Scene.cpp
)

set(
  HEADERS
  
  include/Scene.h
)

add_library(
  Scene
  
  ${SOURCES}
  ${HEADERS}
)

target_include_directories(Scene PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(Scene PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/py")

target_link_libraries(Scene PRIVATE Math)
target_link_libraries(Scene PRIVATE Common)
target_link_libraries(Scene PRIVATE Visual)
target_link_libraries(Scene PRIVATE Primitives)