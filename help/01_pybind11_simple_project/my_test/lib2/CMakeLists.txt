set(
    headers
    
    include/lib2.h    
)

set(
    sources
    
    src/lib2.cpp
)

add_library( 
    lib2
    
    ${sources}
    ${headers}
)

target_include_directories(lib2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(lib2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/py")
