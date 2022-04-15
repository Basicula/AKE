#include <ILight_py.hpp>
#include <SpotLight_py.hpp>

static void AddLightSubmodule(py::module& io_module)
  {
  auto light_submodule = io_module.def_submodule("Light");
  AddILight(light_submodule);
  AddSpotLight(light_submodule);
  }