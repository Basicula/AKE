#include <ISurface.h>

namespace
  {
  class PyISurface : public ISurface
    {
    using ISurface::ISurface;
    BoundingBox _GetBoundingBox() const override
      {
      PYBIND11_OVERLOAD_PURE(
        BoundingBox,
        ISurface,
        _GetBoundingBox, );
      }

    bool _IntersectWithRay(
      IntersectionRecord& o_intersection,
      const Ray& i_ray) const override
      {
      PYBIND11_OVERLOAD_PURE(
        bool,
        ISurface,
        _IntersectWithRay,
        o_intersection,
        i_ray);
      }

    Vector3d _NormalAtPoint(const Vector3d& i_point) const override
      {
      PYBIND11_OVERLOAD_PURE(
        Vector3d,
        ISurface,
        _NormalAtPoint,
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
      auto primitives_m = py::module::import("engine.Primitives");
      if (i_dict.contains("Sphere"))
        return primitives_m
          .attr("Sphere")
          .attr("fromDict")(i_dict)
          .cast<ISurfaceSPtr>();
      return ISurfaceSPtr();
      });
  }