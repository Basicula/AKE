#include <IMaterial_py.hpp>
#include <ColorMaterial_py.hpp>

static void AddMaterialSubmodule(py::module& io_module)
  {
  auto material_submodule = io_module.def_submodule("Material");
  AddIMaterial(material_submodule);
  AddColorMaterial(material_submodule);
  }