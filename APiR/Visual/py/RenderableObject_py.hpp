#include <RenderableObject.h>

namespace
  {
  class PyRenderableObject : public RenderableObject
    {
    using RenderableObject::RenderableObject;
    bool IntersectWithRay(
      IntersectionRecord& io_intersection,
      const Ray& i_ray) const override
      {
      PYBIND11_OVERLOAD(
        bool,
        RenderableObject,
        IntersectWithRay,
        io_intersection,
        i_ray);
      }

    BoundingBox GetBoundingBox() const override
      {
      PYBIND11_OVERLOAD(
        BoundingBox,
        RenderableObject,
        GetBoundingBox, );
      }

    std::string Serialize() const override
      {
      PYBIND11_OVERLOAD(
        std::string,
        RenderableObject,
        Serialize, );
      }
    };
  }

static void AddRenderableObject(py::module& io_module)
  {
  py::class_<
    RenderableObject, 
    std::shared_ptr<RenderableObject>,
    IRenderable,
    PyRenderableObject>(io_module, "RenderableObject")
    .def(py::init<ISurfaceSPtr, IMaterialSPtr>(),
      py::arg("surface"), py::arg("material"))
    .def("fromDict", [](py::dict i_dict)
      {
      auto primitives_m = py::module::import("engine.Primitives");
      auto material_m = py::module::import("engine.Visual.Material");
      auto inner = i_dict["RenderableObject"];
      auto surface = primitives_m
        .attr("ISurface")
        .attr("fromDict")(inner["Surface"])
        .cast<ISurfaceSPtr>();
      auto material = material_m
        .attr("IMaterial")
        .attr("fromDict")(inner["Material"])
        .cast<IMaterialSPtr>();
      return RenderableObject(surface, material);
      });
  }