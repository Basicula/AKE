#include <IRenderable.h>

namespace
  {
  class PyIRenderable : public IRenderable
    {
    using IRenderable::IRenderable;
    bool IntersectWithRay(
      IntersectionRecord& io_intersection,
      const Ray& i_ray) const override
      {
      PYBIND11_OVERLOAD_PURE(
        bool,
        IRenderable,
        IntersectWithRay,
        io_intersection,
        i_ray);
      }

    BoundingBox GetBoundingBox() const override
      {
      PYBIND11_OVERLOAD_PURE(
        BoundingBox,
        IRenderable,
        GetBoundingBox, );
      }
    };
  }

static void AddIRenderable(py::module& io_module)
  {
  py::class_<
    IRenderable,
    std::shared_ptr<IRenderable>,
    IObject,
    PyIRenderable>(io_module, "IRenderable")
    .def("hitRay", &IRenderable::IntersectWithRay)
    .def_property_readonly("boundingBox", &IRenderable::GetBoundingBox)
    .def("fromDict", [](py::dict i_dict)
      {
      auto renderable_m = py::module::import("engine.Visual.Renderable");
      if (i_dict.contains("RenderableObject"))
        return renderable_m
          .attr("RenderableObject")
          .attr("fromDict")(i_dict)
          .cast<IRenderableSPtr>();
      return IRenderableSPtr();
      });
  }