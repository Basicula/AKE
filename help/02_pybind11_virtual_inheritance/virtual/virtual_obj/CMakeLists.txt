set(
    headers
    
    include/IObject.h    
    include/Object.h    
)

set(
    sources
    
    src/Object.cpp
)

add_library( 
    virtual_obj
    
    ${sources}
    ${headers}
)

target_include_directories(virtual_obj PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(virtual_obj PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/py")

