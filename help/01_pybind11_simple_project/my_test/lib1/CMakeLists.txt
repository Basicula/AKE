set(
    headers
    
    include/lib1.h    
)

set(
    sources
    
    src/lib1.cpp
)

add_library( 
    lib1
    
    ${sources}
    ${headers}
)

target_include_directories(lib1 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(lib1 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/py")

