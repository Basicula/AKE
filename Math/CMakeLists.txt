add_library(
	Math 
	
	src/SolveEquations.cpp
	
	include/SolveEquations.h
	include/Vector.h

	include/impl/VectorImpl.h
)

target_include_directories(Math PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(Math PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/impl")