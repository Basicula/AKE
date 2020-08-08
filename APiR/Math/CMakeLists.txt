set(
  SOURCES
    
  src/SolveEquations.cpp
  
  src/Matrix.cpp
)

set(
  HEADERS
    
  include/SolveEquations.h
  
  include/Vector.h
  include/impl/VectorImpl.h
  include/impl/Vector2dImpl.h
  include/impl/Vector3dImpl.h
  
  include/Matrix.h
  include/Matrix3.h
  include/impl/Matrix3Impl.h
)

set(
  TESTS

  test/c++/Vector3d_test.cpp
  
  test/c++/Matrix3d_test.cpp
)

add_library(
  Math	
    
  ${SOURCES}
  ${HEADERS}
)

target_include_directories(Math PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(Math PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/impl")

target_include_directories(Math PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/py")

# Test library
add_executable(Math.Test ${TESTS})
target_link_libraries(Math.Test PRIVATE Math)
target_link_libraries(Math.Test PRIVATE gtest_main)
add_test(
  NAME
    unit
  COMMAND
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/Math.Test
  )