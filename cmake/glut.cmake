find_package(GLUT QUIET)

if(NOT GLUT_FOUND)
	message("Installing GLUT...")
	
	set(FreeglutPath "${THIRD_PARTIES}/freeglut_mvc")
	set(GLUT_INCLUDE_DIRS "${FreeglutPath}/include" PARENT_SCOPE)
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(GLUT_LIBRARY_DIRS "${FreeglutPath}/lib/x64" PARENT_SCOPE)
		file(	COPY "${FreeglutPath}/bin/x64/freeglut.dll"
			DESTINATION "${CMAKE_BINARY_DIR}"
		)
	else()
		set(GLUT_LIBRARY_DIRS "${FreeglutPath}/lib" PARENT_SCOPE)
		file(	COPY "${FreeglutPath}/bin/freeglut.dll"
			DESTINATION "${CMAKE_BINARY_DIR}"
		)
	endif()
endif()