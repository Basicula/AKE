find_package(OpenCL QUIET) # check if already exists on pc
# init chache variables for cmake to correct OpenCL processing
if(NOT OpenCL_FOUND)
  message("OpenCL has not be found, using OpenCL from ThirdParties")
  set(OpenCLPath "${THIRD_PARTIES}/OpenCL" CACHE PATH "" FORCE)
  set(OpenCL_INCLUDE_DIR "${OpenCLPath}/include" CACHE PATH "" FORCE)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(OpenCL_LIBRARY "${OpenCLPath}/lib/x64/OpenCL.lib" CACHE PATH "" FORCE)
	else()
		set(OpenCL_LIBRARY "${OpenCLPath}/lib/x32/OpenCL.lib" CACHE PATH "" FORCE)
	endif()
endif()
find_package(OpenCL REQUIRED) # let CMake do other work
include_directories(${OpenCL_INCLUDE_DIRS}) # include OpenCL dirs after all