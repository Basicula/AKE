#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>

namespace py = pybind11;

#include <math_py.hpp>
#include <common_py.hpp>
#include <visual_py.hpp>
#include <primitives_py.hpp>
#include <scene_module_py.hpp>
#include <fluid_module_py.hpp>

PYBIND11_MODULE(engine, m)
  {
  AddMathSubmodule(m);
  AddCommonSubmodule(m);
  AddPrimitivesSubmodule(m);
  AddVisualSubmodule(m);
  AddFluidSubmodule(m);
  AddSceneSubmodule(m);
  }