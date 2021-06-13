set(
    SOURCES
    src/example.cpp
)

set(
    HEADERS
    include/example.h
)

add_library(example ${SOURCES} ${HEADERS})


target_include_directories(
    example
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )

add_executable(unit_tests test/example_add.cpp test/example_subtract.cpp)

target_link_libraries(unit_tests PRIVATE gtest_main)
target_link_libraries(unit_tests PRIVATE example)

add_test(
  NAME
    unit
  COMMAND
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unit_tests
  )
