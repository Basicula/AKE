#include <ColorMaterial.h>

namespace
  {
  class PyColorMaterial : public ColorMaterial
    {
    using ColorMaterial::ColorMaterial;
    std::string Serialize() const override
      {
      PYBIND11_OVERLOAD(
        std::string,
        ColorMaterial,
        Serialize,
        );
      }
    Color GetPrimitiveColor() const override
      {
      PYBIND11_OVERLOAD(
        Color,
        ColorMaterial,
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
        ColorMaterial,
        GetLightInfluence,
        i_point,
        i_normal,
        i_view_direction,
        i_light
      );
      }
    };
  }

static void AddColorMaterial(py::module& io_module)
  {
  py::class_<ColorMaterial, std::shared_ptr<ColorMaterial>, IMaterial, PyColorMaterial>(io_module, "ColorMaterial")
    .def(py::init<Color,
      const Vector3d&,
      const Vector3d&,
      const Vector3d&,
      double,
      double,
      double>(),
      py::arg("color"),
      py::arg("ambient") = Vector3d(1.0, 1.0, 1.0),
      py::arg("diffuse") = Vector3d(1.0, 1.0, 1.0),
      py::arg("specular") = Vector3d(1.0, 1.0, 1.0),
      py::arg("shinines") = 1.0,
      py::arg("reflection") = 0.0,
      py::arg("refraction") = 0.0)
    .def_property("color",
      &ColorMaterial::GetColor,
      &ColorMaterial::SetColor)
    .def_property("ambient",
      &ColorMaterial::GetAmbient,
      &ColorMaterial::SetAmbient)
    .def_property("diffuse",
      &ColorMaterial::GetDiffuse,
      &ColorMaterial::SetDiffuse)
    .def_property("specular",
      &ColorMaterial::GetSpecular,
      &ColorMaterial::SetSpecular)
    .def_property("reflection",
      &ColorMaterial::GetReflection,
      &ColorMaterial::SetReflection)
    .def_property("refraction",
      &ColorMaterial::GetRefraction,
      &ColorMaterial::SetRefraction)
    .def_property("shinines",
      &ColorMaterial::GetShinines,
      &ColorMaterial::SetShinines)
    .def("acolor", &ColorMaterial::GetAmbientColor)
    .def("dcolor", &ColorMaterial::GetDiffuseColor)
    .def("scolor", &ColorMaterial::GetSpecularColor)
    .def("fromDict", [](py::dict i_dict)
      {
      auto vec_m = py::module::import("engine.Math.Vector");
      auto color_m = py::module::import("engine.Visual");
      auto inner = i_dict["ColorMaterial"];
      Color color = color_m.attr("Color").attr("fromDict")(inner["Color"]).cast<Color>();
      Vector3d ambient = vec_m.attr("Vector3d").attr("fromDict")(inner["Ambient"]).cast<Vector3d>();
      Vector3d diffuse = vec_m.attr("Vector3d").attr("fromDict")(inner["Diffuse"]).cast<Vector3d>();
      Vector3d specular = vec_m.attr("Vector3d").attr("fromDict")(inner["Specular"]).cast<Vector3d>();
      double shinines = inner["Shinines"].cast<double>();
      double reflection = inner["Reflection"].cast<double>();
      double refraction = inner["Refraction"].cast<double>();
      return ColorMaterial(
        color,
        ambient,
        diffuse,
        specular,
        shinines,
        reflection,
        refraction);
      });
  }