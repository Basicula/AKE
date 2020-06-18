set(
  SOURCES
    
	src/Sphere.cpp
	src/Plane.cpp
	src/Torus.cpp
	src/Cylinder.cpp
)

set(
  HEADERS
  
  include/ISurface.h
  
	include/Sphere.h
	include/Plane.h
	include/Torus.h
	include/Cylinder.h
)

add_library(
	Primitives
    
  ${SOURCES}
  ${HEADERS}
)

target_include_directories(Primitives PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(Primitives PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/py")

target_link_libraries(Primitives PRIVATE Math)
target_link_libraries(Primitives PRIVATE Common)