add_library(
	BMPWriter
	src/BMPWriter.cpp
	src/BMPUtilities.cpp

	includes/BMPWriter.h
	includes/BMPUtilities.h
)

target_include_directories(BMPWriter PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/includes")