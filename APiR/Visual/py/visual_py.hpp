#include <Camera_py.hpp>
#include <Color_py.hpp>
#include <Image_py.hpp>

#include <material_py.hpp>
#include <light_py.hpp>
#include <renderable_py.hpp>

static void AddVisualSubmodule(py::module& io_module)
  {
  auto visual_submodule = io_module.def_submodule("Visual");
  AddCamera(visual_submodule);
  AddColor(visual_submodule);
  AddImage(visual_submodule);

  AddMaterialSubmodule(visual_submodule);
  AddLightSubmodule(visual_submodule);
  AddRenderableSubmodule(visual_submodule);
  }