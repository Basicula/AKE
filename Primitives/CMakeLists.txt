add_library(
	Primitives
	
	src/IntersectionUtilities.cpp
	src/Sphere.cpp
	src/Ray.cpp
	src/Plane.cpp
	src/Torus.cpp
	src/Cylinder.cpp

	include/IntersectionUtilities.h
	include/Sphere.h
	include/Ray.h
	include/Plane.h
	include/Torus.h
	include/Cylinder.h

	include/IObject.h
)

target_include_directories(Primitives PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(Primitives PRIVATE Math)
target_link_libraries(Primitives PRIVATE VisualAspects)