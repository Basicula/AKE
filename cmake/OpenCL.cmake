find_package(OpenCL QUIET)

if(NOT GLUT_FOUND)
	message("Installing OpenCL...")
	
	set(OpenCLPath "${THIRD_PARTIES}/OpenCL")
	
	set(OPENCL_INCLUDE_DIRS "${OpenCLPath}/include" PARENT_SCOPE)
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(OPENCL_LIBRARY_DIRS "${OpenCLPath}/lib/x64" PARENT_SCOPE)
		#file(	COPY "${OpenCLPath}/bin/x64/.dll"
		#	DESTINATION "${CMAKE_BINARY_DIR}"
		#)
	else()
		set(OPENCL_LIBRARY_DIRS "${OpenCLPath}/lib/x32" PARENT_SCOPE)
		#file(	COPY "${OpenCLPath}/bin/.dll"
		#	DESTINATION "${CMAKE_BINARY_DIR}"
		#)
	endif()
endif()