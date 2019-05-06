cmake_minimum_required(VERSION 3.12)

project(RayTracing)

add_subdirectory(BMPWriter)
add_subdirectory(Math)
add_subdirectory(Primitives)
add_subdirectory(VisualAspects)

#OpenGL

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
link_directories(${OPENGL_LIBRARY_DIRS})

#OpenGL end

#GLUT

find_package(GLUT QUIET)

if(NOT GLUT_FOUND)
	message("Installing GLUT...")
	
	set(FreeglutPath ThirdParties/freeglut_mvc)
	
	set(GLUT_INCLUDE_DIRS ${FreeglutPath}/include)
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(GLUT_LIBRARY_DIRS ${FreeglutPath}/lib/x64)
		file(	COPY ${FreeglutPath}/bin/x64/freeglut.dll
			DESTINATION ${CMAKE_BINARY_DIR}
		)
	else()
		set(GLUT_LIBRARY_DIRS ${FreeglutPath}/lib)
		file(	COPY ${FreeglutPath}/bin/freeglut.dll
			DESTINATION ${CMAKE_BINARY_DIR}
		)
	endif()
	
	link_directories(${GLUT_LIBRARY_DIRS})
endif()

include_directories(${GLUT_INCLUDE_DIRS})

#GLUT end

add_executable(RayTracing main.cpp)

target_link_libraries(RayTracing PRIVATE BMPWriter)
target_link_libraries(RayTracing PRIVATE Math)
target_link_libraries(RayTracing PRIVATE Primitives)
target_link_libraries(RayTracing PRIVATE VisualAspects)

#ThirdParties
target_link_libraries(RayTracing PRIVATE ${OPENGL_LIBRARIES})
target_link_libraries(RayTracing PRIVATE freeglut)