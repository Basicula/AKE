add_subdirectory(Math)
add_subdirectory(OpenCLKernel)
add_subdirectory(Common)
add_subdirectory(Visual)
add_subdirectory(BMPWriter)
add_subdirectory(Primitives)
add_subdirectory(Fluid)
add_subdirectory(Scene)
add_subdirectory(GLUTWindow)

pybind11_add_module(engine engine_module.cpp)
target_link_libraries(
    engine
    
    PRIVATE
    
    Math
    Common
    Visual
    Primitives
    Fluid
    Scene
    
    pybind11)
    
set_target_properties(Primitives PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Common PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Fluid PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Visual PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Scene PROPERTIES POSITION_INDEPENDENT_CODE TRUE)


if (WIN32)
  add_executable(EXE main.cpp)
  
  target_link_libraries(EXE PRIVATE Math)
  target_link_libraries(EXE PRIVATE Common)
  target_link_libraries(EXE PRIVATE Visual)
  target_link_libraries(EXE PRIVATE Primitives)
  target_link_libraries(EXE PRIVATE Fluid)
  target_link_libraries(EXE PRIVATE Scene)
  target_link_libraries(EXE PRIVATE GLUTWindow)
  target_link_libraries(EXE PRIVATE BMPWriter)
endif()