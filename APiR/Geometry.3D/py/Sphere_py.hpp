#include <Geometry/Sphere.h>

namespace
  {
  class PySphere : public Sphere
    {
    using Sphere::Sphere;
    bool _IntersectWithRay(
      double& io_nearest_intersection_dist,
      const Ray& i_ray) const override
      {
      PYBIND11_OVERLOAD(
        bool,
        Sphere,
        _IntersectWithRay,
        io_nearest_intersection_dist,
        i_ray);
      }
    std::string Serialize() const override
      {
      PYBIND11_OVERLOAD(
        std::string,
        Sphere,
        Serialize,);
      }
    void _CalculateBoundingBox() override
      {
      PYBIND11_OVERLOAD(
        void,
        Sphere,
        _CalculateBoundingBox,);
      }
    };
  }

static void AddSphere(py::module& io_module)
  {
  py::class_<
    Sphere, 
    std::shared_ptr<Sphere>, 
    ISurface, 
    PySphere>(io_module, "Sphere")
    .def(py::init<
      const Vector3d&,
      double>(),
      py::arg("center"),
      py::arg("radius"))
    .def_property("center",
      &Sphere::GetCenter,
      &Sphere::SetCenter)
    .def_property("radius",
      &Sphere::GetRadius,
      &Sphere::SetRadius)
    .def("fromDict", [](py::dict i_dict)
      {
      auto vec_m = py::module::import("engine.Math.Vector");
      auto inner = i_dict["Sphere"];
      Vector3d center = vec_m
        .attr("Vector3d")
        .attr("fromDict")(inner["Center"])
        .cast<Vector3d>();
      double radius = inner["Radius"].cast<double>();
      return Sphere(center,radius);
      });
  }