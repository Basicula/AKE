#include <Fluid/Fluid.h>

namespace
  {
  class PyFluid : public Fluid
    {
    using Fluid::Fluid;
    bool IntersectWithRay(
      IntersectionRecord& io_intersection,
      const Ray& i_ray) const override
      {
      PYBIND11_OVERLOAD(
        bool,
        Fluid,
        IntersectWithRay,
        io_intersection,
        i_ray);
      }
    std::string Serialize() const override
      {
      PYBIND11_OVERLOAD(
        std::string,
        Fluid,
        Serialize, );
      }
    BoundingBox GetBoundingBox() const override
      {
      PYBIND11_OVERLOAD(
        BoundingBox,
        Fluid,
        GetBoundingBox, );
      }
    };
  }

static void AddFluid(py::module& io_module)
  {
  py::class_<
    Fluid,
    std::shared_ptr<Fluid>,
    IRenderable,
    PyFluid>(io_module, "Fluid")
    .def(py::init<
      std::size_t>(),
      py::arg("numOfParticles"))
    .def_property_readonly("numOfParticles", &Fluid::GetNumParticles)
    .def("update", &Fluid::Update)
    .def("fromDict", [](py::dict i_dict)
      {
      auto common_m = py::module::import("engine.Common");
      auto inner = i_dict["Fluid"];
      std::size_t num_of_particles =
        inner["NumOfParticles"].cast<std::size_t>();
      return Fluid(num_of_particles);
      });
  }