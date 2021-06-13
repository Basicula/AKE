add_subdirectory(virtual_obj)

pybind11_add_module(virtual virtual.cpp)
target_link_libraries(
    virtual
    
    PRIVATE
    
    virtual_obj
    
    pybind11)
    
set_target_properties(virtual_obj PROPERTIES POSITION_INDEPENDENT_CODE TRUE)