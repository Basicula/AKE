add_library(
	Primitives
	
	src/Sphere.cpp
	src/Ray.cpp
	src/IntersectionUtilities.cpp
	
	include/Sphere.h
	include/Ray.h
	include/IntersectionUtilities.h
)

target_include_directories(Primitives PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(Primitives PRIVATE Math)