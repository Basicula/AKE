add_library(
	VisualAspects
	
	src/SpotLight.cpp
	src/Color.cpp
	src/ColorMaterial.cpp
	
	include/SpotLight.h
	include/Color.h
	include/ColorMaterial.h
)

target_include_directories(VisualAspects PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(VisualAspects PRIVATE Math)