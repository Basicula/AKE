#include <BoundingBox.h>

static void AddBoundingBox(py::module& io_module)
  {
  py::class_<BoundingBox>(io_module, "BoundingBox")
    .def(py::init<>())
    .def(py::init<
      const Vector3d&,
      const Vector3d&>())
    .def_property_readonly("min", &BoundingBox::GetMin)
    .def_property_readonly("max", &BoundingBox::GetMax)
    .def_property_readonly("center", &BoundingBox::Center)
    .def_property_readonly("deltaX", &BoundingBox::DeltaX)
    .def_property_readonly("deltaY", &BoundingBox::DeltaY)
    .def_property_readonly("deltaZ", &BoundingBox::DeltaZ)
    .def("isValid", &BoundingBox::IsValid)
    .def("addPoint", &BoundingBox::AddPoint)
    .def("contains", &BoundingBox::Contains)
    .def("merge", &BoundingBox::Merge)
    .def("__repr__", &BoundingBox::Serialize)
    .def("fromDict", [](py::dict i_dict)
      {
      auto common_m = py::module::import("engine.Common");
      auto vector_m = py::module::import("engine.Math.Vector");
      auto inner = i_dict["BoundingBox"];
      Vector3d min = vector_m.attr("Vector3d").attr("fromDict")(inner["MinCorner"]).cast<Vector3d>();
      Vector3d max = vector_m.attr("Vector3d").attr("fromDict")(inner["MaxCorner"]).cast<Vector3d>();
      return BoundingBox(min,max);
      });
  }