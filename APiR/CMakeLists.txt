if (ENABLE_OPENCL)
add_subdirectory(OpenCLCore)
add_subdirectory(OpenCLKernels)
endif()
if (ENABLE_CUDA)
add_subdirectory(CUDACore)
endif()
add_subdirectory(Memory)
add_subdirectory(Common)
add_subdirectory(Math)
add_subdirectory(Image)
add_subdirectory(Fractal)
add_subdirectory(Rendering)
add_subdirectory(Rendering.2D)
add_subdirectory(Visual)
add_subdirectory(BMPWriter)
add_subdirectory(Physics)
add_subdirectory(Geometry)
add_subdirectory(Geometry.2D)
add_subdirectory(Geometry.3D)
add_subdirectory(Fluid)
add_subdirectory(Window)
add_subdirectory(Main)

if(ENABLE_PYTHON_WRAPPER)
  pybind11_add_module(engine engine_module.cpp)
  target_link_libraries(
    engine
    
    PRIVATE
    
    Common
    Fluid
    Geometry
    Image
    Math
    Memory
    Physics
    Rendering
    Visual
    
    pybind11
  )
  set_property(TARGET engine PROPERTY FOLDER "APiR")
endif()

set_target_properties(Common PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Fluid PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Geometry PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Math PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Physics PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Rendering PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(Visual PROPERTIES POSITION_INDEPENDENT_CODE TRUE)