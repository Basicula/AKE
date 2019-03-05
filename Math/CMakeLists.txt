add_library(Math INTERFACE)

target_include_directories(Math INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(Math INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")