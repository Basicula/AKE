#include "Geometry.3D/ISurface.h"

namespace
  {
  class PyISurface : public ISurface
    {
    using ISurface::ISurface;
    void _CalculateBoundingBox() override
      {
      PYBIND11_OVERLOAD_PURE(
        void,
        ISurface,
        _CalculateBoundingBox, );
      }

    bool _IntersectWithRay(
      double& io_nearest_intersection_dist,
      const Ray& i_ray) const override
      {
      PYBIND11_OVERLOAD_PURE(
        bool,
        ISurface,
        _IntersectWithRay,
        io_nearest_intersection_dist,
        i_ray);
      }

    Vector3d _NormalAtLocalPoint(const Vector3d& i_point) const override
      {
      PYBIND11_OVERLOAD_PURE(
        Vector3d,
        ISurface,
        _NormalAtLocalPoint,
        i_point);
      }

    std::string Serialize() const override
      {
      PYBIND11_OVERLOAD_PURE(
        std::string,
        ISurface,
        Serialize,
        );
      }
    };
  }

static void AddISurface(py::module& io_module)
  {
  py::class_<
    ISurface, 
    ISurfaceSPtr, 
    PyISurface>(io_module, "ISurface")
    .def_property_readonly("boundingBox", &ISurface::GetBoundingBox)
    .def("hitRay", &ISurface::IntersectWithRay)
    .def("normalAtPoint", &ISurface::NormalAtPoint)
    .def("__repr__", &ISurface::Serialize)
    .def("fromDict", [](py::dict i_dict)
      {
      auto primitives_m = py::module::import("engine.Geometry");
      if (i_dict.contains("Sphere"))
        return primitives_m
          .attr("Sphere")
          .attr("fromDict")(i_dict)
          .cast<ISurfaceSPtr>();
      return ISurfaceSPtr();
      });
  }