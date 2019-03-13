add_library(
	BMPWriter
	src/BMPWriter.cpp
	src/BMPUtilities.cpp

	include/BMPWriter.h
	include/BMPUtilities.h
)

target_include_directories(BMPWriter PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(BMPWriter PUBLIC VisualAspects)