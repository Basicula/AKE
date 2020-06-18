#include <Intersection.h>

namespace
  {
  static void AddIntersectionRecord(py::module& io_module)
    {
    py::class_<IntersectionRecord>(io_module, "IntersectionRecord")
      .def(py::init<>())
      .def_readonly("intersection", &IntersectionRecord::m_intersection)
      .def_readonly("normal", &IntersectionRecord::m_normal)
      .def_readonly("distance", &IntersectionRecord::m_distance)
      .def_readonly("material", &IntersectionRecord::m_material);
    }

  static void AddRayBoxIntersectionRecord(py::module& io_module)
    {
    py::class_<RayBoxIntersectionRecord>(io_module, "RayBoxIntersectionRecord")
      .def(py::init<>())
      .def_readonly("intersected", &RayBoxIntersectionRecord::m_intersected)
      .def_readonly("tmin", &RayBoxIntersectionRecord::m_tmin)
      .def_readonly("tmax", &RayBoxIntersectionRecord::m_tmax);
    }

  static void AddIntersectionUtils(py::module& io_module)
    {
    auto intersection_utils = io_module.def_submodule("Utils");
    intersection_utils.def("rayIntersectBox", RayIntersectBox);
    intersection_utils.def("rayBoxIntersection", RayBoxIntersection);
    }
  }

static void AddIntersectionSubmodule(py::module& io_module)
  {
  auto intersection_m = io_module.def_submodule("Intersection");
  AddIntersectionRecord(intersection_m);
  AddRayBoxIntersectionRecord(intersection_m);
  AddIntersectionUtils(intersection_m);
  }