#include <IObject_py.hpp>

static void AddCommonSubmodule(py::module& io_module)
  {
  auto common_submodule = io_module.def_submodule("Common");
  AddIObject(common_submodule);
  }