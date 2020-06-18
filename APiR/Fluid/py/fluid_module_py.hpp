#include <Fluid_py.hpp>

static void AddFluidSubmodule(py::module& io_module)
  {
  auto fluid_submodule = io_module.def_submodule("Fluid");

  AddFluid(fluid_submodule);
  }