#include <Common/IObject.h>

namespace
  {
  class PyIObject : public IObject
    {
    using IObject::IObject;
    std::string Serialize() const override
      {
      PYBIND11_OVERLOAD_PURE(
        std::string,
        IObject,
        Serialize,
        );
      }
    };
  }

static void AddIObject(py::module& io_module)
  {
  py::class_<IObject, std::shared_ptr<IObject>, PyIObject>(io_module, "IObject")
    .def("__repr__", &IObject::Serialize)
    .def("fromDict", [](py::dict i_dict)
      {
      auto primitives_m = py::module::import("engine.Geometry");
      if (i_dict.contains("Sphere"))
        return primitives_m.attr("Sphere").attr("fromDict")(i_dict).cast<std::shared_ptr<IObject>>();
      return std::shared_ptr<IObject>();
      });
  }