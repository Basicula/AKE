set(
  SOURCES
    
	src/SolveEquations.cpp
	src/TransformationMatrix.cpp
)

set(
  HEADERS
    
  include/SolveEquations.h
	include/Vector.h
	include/TransformationMatrix.h

	include/impl/VectorImpl.h
	include/impl/Vector3dImpl.h
)

add_library(
	Math	
    
    ${SOURCES}
    ${HEADERS}
)

target_include_directories(Math PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(Math PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/impl")

target_include_directories(Math PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/py")