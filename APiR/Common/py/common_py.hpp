#include <Ray_py.hpp>
#include <BoundingBox_py.hpp>
#include <Intersection_py.hpp>
#include <IObject_py.hpp>

static void AddCommonSubmodule(py::module& io_module)
  {
  auto common_submodule = io_module.def_submodule("Common");
  AddRay(common_submodule);
  AddBoundingBox(common_submodule);
  AddIntersectionSubmodule(common_submodule);
  AddIObject(common_submodule);
  }