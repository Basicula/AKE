add_library(
	BMPWriter
	src/BMPWriter.cpp
	includes/BMPWriter.h
)

target_include_directories(BMPWriter PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/includes")