set(
    SOURCES
    
    src/BoundingBox.cpp
    src/Intersection.cpp
    src/Ray.cpp
    
    src/BFPointSearcher.cpp
    
    src/Simulation.cpp
    
    src/ThreadPool.cpp
)

set(
    HEADERS
    
    include/BoundingBox.h
    include/Intersection.h
    include/Ray.h

    include/PointNeighborSearcher.h
    include/BFPointSearcher.h
    
    include/Simulation.h
    
    include/ThreadPool.h
    include/impl/ThreadPool_impl.h
    include/ParallelUtils.h
    
    include/IObject.h
    
    include/DefinesAndConstants.h
)

set(
    TESTS
    
    test/c++/ThreadPool_test.cpp
    
    test/c++/Utils.h
)

# Main library
add_library(
    Common
    
    ${SOURCES}
    ${HEADERS}
)

target_include_directories(Common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(Common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/impl")

target_include_directories(Common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/py")

target_link_libraries(Common PRIVATE Math)

# Test library
add_executable(Common.Test ${TESTS})
target_link_libraries(Common.Test PRIVATE Common)
target_link_libraries(Common.Test PRIVATE gtest_main)
add_test(
  NAME
    unit
  COMMAND
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/Common.Test
  )
