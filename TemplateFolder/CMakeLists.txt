add_library(
	Example
	src/Example.cpp

	include/Example.h
)

target_include_directories(Example PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(Example PRIVATE smth)