#include <BoundingBox_py.hpp>
#include <ISurface_py.hpp>
#include <Sphere_py.hpp>
#include <Ray_py.hpp>
#include <Intersection_py.hpp>

static void AddGeometrySubmodule(py::module& io_module)
  {
  auto geometry_submodule = io_module.def_submodule("Geometry");

  AddISurface(geometry_submodule);
  AddSphere(geometry_submodule);

  AddBoundingBox(geometry_submodule);
  AddRay(geometry_submodule);

  AddIntersectionSubmodule(geometry_submodule);
  }