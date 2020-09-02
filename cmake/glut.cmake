# GLUT_INCLUDE_DIR, where to find GL/glut.h, etc.
# GLUT_LIBRARIES, the libraries to link against
# GLUT_FOUND, If false, do not try to use GLUT.
# Also defined, but not for general use are:
# 
# GLUT_glut_LIBRARY = the full path to the glut library.
# GLUT_Xmu_LIBRARY  = the full path to the Xmu library.
# GLUT_Xi_LIBRARY   = the full path to the Xi Library.

if(NOT GLUT_FOUND)
  find_package(GLUT QUIET)
endif()

if(NOT GLUT_FOUND)
	message("FreeGlut has not be found, using it from ThirdParties")
	set(GLUT_FOUND ON CACHE BOOL "")
	set(GLUT_glut_LIBRARY "${THIRD_PARTIES}/freeglut_mvc" CACHE PATH "")
	set(GLUT_INCLUDE_DIR "${GLUT_glut_LIBRARY}/include" CACHE PATH "")
  
	set(GLUT_INCLUDE_DIRS "${GLUT_glut_LIBRARY}/include" CACHE PATH "")
  set(GLUT_LIBRARIES freeglut.lib)
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(GLUT_LIBRARY_DIRS "${GLUT_glut_LIBRARY}/lib/x64" CACHE PATH "")
		set(GLUT_BINARY "${GLUT_glut_LIBRARY}/bin/x64/freeglut.dll" CACHE PATH "")
	else()
		set(GLUT_LIBRARY_DIRS "${GLUT_glut_LIBRARY}/lib/x32" CACHE PATH "")
		set(GLUT_BINARY "${GLUT_glut_LIBRARY}/bin/x32/freeglut.dll" CACHE PATH "")
	endif()
endif()

include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})