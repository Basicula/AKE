cmake_minimum_required(VERSION 3.12)

project(RayTracing)

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

#OpenCL

find_package(OpenCL QUIET)

if(NOT GLUT_FOUND)
	message("Installing OpenCL...")
	
	set(OpenCLPath ThirdParties/OpenCL)
	
	set(OPENCL_INCLUDE_DIRS ${OpenCLPath}/include)
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(OPENCL_LIBRARY_DIRS ${OpenCLPath}/lib/x64)
		#file(	COPY ${OpenCLPath}/bin/x64/.dll
		#	DESTINATION ${CMAKE_BINARY_DIR}
		#)
	else()
		set(OPENCL_LIBRARY_DIRS ${OpenCLPath}/lib/x32)
		#file(	COPY ${OpenCLPath}/bin/.dll
		#	DESTINATION ${CMAKE_BINARY_DIR}
		#)
	endif()
	
	link_directories(${OPENCL_LIBRARY_DIRS})
endif()

include_directories(${OPENCL_INCLUDE_DIRS})

#OpenCL end

add_subdirectory(BMPWriter)
add_subdirectory(Math)
add_subdirectory(Primitives)
add_subdirectory(VisualAspects)
add_subdirectory(GLUTWindow)
add_subdirectory(OpenCLKernel)

add_executable(RayTracing main.cpp)

#ThirdParties
#target_link_libraries(RayTracing PRIVATE ${OPENGL_LIBRARIES})
#target_link_libraries(RayTracing PRIVATE freeglut)
target_link_libraries(RayTracing PUBLIC OpenCL)

target_link_libraries(RayTracing PUBLIC BMPWriter)
target_link_libraries(RayTracing PUBLIC Math)
target_link_libraries(RayTracing PUBLIC Primitives)
target_link_libraries(RayTracing PUBLIC VisualAspects)
target_link_libraries(RayTracing PUBLIC GLUTWindow)
target_link_libraries(RayTracing PUBLIC OpenCLKernel)
