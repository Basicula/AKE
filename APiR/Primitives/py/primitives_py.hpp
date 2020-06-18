#include <ISurface_py.hpp>
#include <Sphere_py.hpp>

static void AddPrimitivesSubmodule(py::module& io_module)
  {
  auto primitives_submodule = io_module.def_submodule("Primitives");

  AddISurface(primitives_submodule);
  AddSphere(primitives_submodule);
  }