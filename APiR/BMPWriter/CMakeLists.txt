add_library(
	BMPWriter
	src/BMPWriter.cpp

	include/BMPWriter.h
)

target_include_directories(BMPWriter PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(BMPWriter PUBLIC Math)
target_link_libraries(BMPWriter PUBLIC Visual)