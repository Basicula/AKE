add_subdirectory(CUDACore)
add_subdirectory(Common)
add_subdirectory(Math)
add_subdirectory(Fractal)
add_subdirectory(Rendering)
add_subdirectory(Visual)
add_subdirectory(BMPWriter)
add_subdirectory(Physics)
add_subdirectory(Geometry)
add_subdirectory(Fluid)
add_subdirectory(OpenCLCore)
add_subdirectory(Scene)
add_subdirectory(GLUTWindow)
add_subdirectory(Main)

add_subdirectory(CUDATest)

pybind11_add_module(engine engine_module.cpp)
target_link_libraries(
  engine
  
  PRIVATE
  
  CUDACore
  Common
  Fluid
  Geometry
  Math
  Physics
  Rendering
  Scene
  Visual
  
  pybind11
)
set_property(TARGET engine PROPERTY FOLDER "APiR")

set_target_properties(Common PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Fluid PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Geometry PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Math PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Physics PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Rendering PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Scene PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Visual PROPERTIES POSITION_INDEPENDENT_CODE TRUE)