add_subdirectory(lib1)
add_subdirectory(lib2)

pybind11_add_module(my_test my_test.cpp)
target_link_libraries(
    my_test
    
    PRIVATE
    
    lib1
    lib2
    
    pybind11)