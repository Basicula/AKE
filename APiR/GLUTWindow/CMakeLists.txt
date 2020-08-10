set(
  SOURCES
  
	src/GLUTWindow.cpp
	src/FPSCounter.cpp
)

set(
  HEADERS
  
	include/GLUTWindow.h
	include/FPSCounter.h
)

add_library(
  GLUTWindow

  ${SOURCES}
  ${HEADERS}
)

target_include_directories(GLUTWindow PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(GLUTWindow PUBLIC ${GLUT_LIBRARIES})

target_link_libraries(GLUTWindow PRIVATE Visual)
target_link_libraries(GLUTWindow PRIVATE Math)