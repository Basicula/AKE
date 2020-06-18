set(
  SOURCES
  
)

set(
  HEADERS
  
)

add_library(
  __temp
  
  ${SOURCES}
  ${HEADERS}
)

target_include_directories(__temp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(__temp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/py")

target_link_libraries(__temp PRIVATE __another)