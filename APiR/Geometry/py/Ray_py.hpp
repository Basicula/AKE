#include <Geometry/Ray.h>

static void AddRay(py::module& io_module)
  {
  py::class_<Ray>(io_module, "Ray")
    .def(py::init<
      const Vector3d&,
      const Vector3d&>())
    .def_property("origin",
      &Ray::GetOrigin,
      &Ray::SetOrigin)
    .def_property("direction",
      &Ray::GetDirection,
      &Ray::SetDirection);
  }