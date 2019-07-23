add_library(
	GLUTWindow
	src/GLUTWindow.cpp

	include/GLUTWindow.h
)

target_include_directories(GLUTWindow PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(GLUTWindow PUBLIC ${OPENGL_LIBRARIES})
target_link_libraries(GLUTWindow PUBLIC freeglut)
target_link_libraries(GLUTWindow PUBLIC VisualAspects)
target_link_libraries(GLUTWindow PUBLIC Math)