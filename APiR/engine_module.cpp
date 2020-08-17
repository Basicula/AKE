#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>

namespace py = pybind11;

#include <common_py.hpp>
#include <fluid_module_py.hpp>
#include <geometry_py.hpp>
#include <math_py.hpp>
#include <rendering_py.hpp>
#include <scene_module_py.hpp>
#include <visual_py.hpp>

PYBIND11_MODULE(engine, m)
  {
  // Add function calls including dependencies
  // i.e. first must be added Rendering and then Fluid 
  // because Fluid depends on Rendering
  AddCommonSubmodule(m);
  AddGeometrySubmodule(m);
  AddMathSubmodule(m);
  AddRenderingSubmodule(m);
  AddSceneSubmodule(m);
  AddVisualSubmodule(m);
  AddFluidSubmodule(m);
  }