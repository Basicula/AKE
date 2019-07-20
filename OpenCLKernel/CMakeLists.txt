add_library(
	OpenCLKernel
	src/OpenCLKernel.cpp

	include/OpenCLKernel.h
)

target_include_directories(OpenCLKernel PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

#target_link_libraries(OpenCLKernel PRIVATE smth)