#include <Vector_py.hpp>

static void AddMathSubmodule(py::module& io_module)
  {
  auto math_submodule = io_module.def_submodule("Math");
  AddVectors(math_submodule);
  }