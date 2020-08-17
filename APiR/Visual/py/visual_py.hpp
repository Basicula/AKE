#include <Color_py.hpp>

#include <material_py.hpp>
#include <light_py.hpp>

static void AddVisualSubmodule(py::module& io_module)
  {
  auto visual_submodule = io_module.def_submodule("Visual");
  AddColor(visual_submodule);

  AddMaterialSubmodule(visual_submodule);
  AddLightSubmodule(visual_submodule);
  }