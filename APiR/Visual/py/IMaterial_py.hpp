#include <IMaterial.h>

namespace
  {
  class PyIMaterial : IMaterial
    {
    using IMaterial::IMaterial;
    std::string Serialize() const override
      {
      PYBIND11_OVERLOAD_PURE(
        std::string,
        IMaterial,
        Serialize,
        );
      }
    Color GetPrimitiveColor() const override
      {
      PYBIND11_OVERLOAD_PURE(
        Color,
        IMaterial,
        GetPrimitiveColor,
        );
      }
    Color GetLightInfluence(
      const Vector3d& i_point,
      const Vector3d& i_normal,
      const Vector3d& i_view_direction,
      std::shared_ptr<ILight> i_light) const override
      {
      PYBIND11_OVERLOAD_PURE(
        Color,
        IMaterial,
        GetLightInfluence,
        i_point,
        i_normal,
        i_view_direction,
        i_light
        );
      }
    };
  }

static void AddIMaterial(py::module& io_module)
  {
  py::class_<IMaterial, std::shared_ptr<IMaterial>, PyIMaterial>(io_module, "IMaterial")
    .def("primitiveColor", &IMaterial::GetPrimitiveColor)
    .def("lightInfluence", &IMaterial::GetLightInfluence)
    .def("__repr__", &IMaterial::Serialize)
    .def("fromDict",[](py::dict i_dict)
      {
      auto material_m = py::module::import("engine.Visual.Material");
      if (i_dict.contains("ColorMaterial"))
        return material_m.attr("ColorMaterial").attr("fromDict")(i_dict).cast<std::shared_ptr<IMaterial>>();
      return std::shared_ptr<IMaterial>();
      });
  }