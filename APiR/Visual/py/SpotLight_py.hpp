#include "Visual/SpotLight.h"

namespace
  {
  class PySpotLight : public SpotLight
    {
    using SpotLight::SpotLight;
    void SetState(bool i_state) override
      {
      PYBIND11_OVERLOAD(
        void,
        SpotLight,
        SetState,
        i_state);
      }
    bool GetState() const override
      {
      PYBIND11_OVERLOAD(
        bool,
        SpotLight,
        GetState,
        );
      }
    std::string Serialize() const override
      {
      PYBIND11_OVERLOAD(
        std::string,
        SpotLight,
        Serialize,
        );
      }
    Vector3d GetDirection(const Vector3d& i_point) const override
      {
      PYBIND11_OVERLOAD(
        Vector3d,
        SpotLight,
        GetDirection,
        i_point);
      }
    };
  }

static void AddSpotLight(py::module io_module)
  {
  py::class_<SpotLight, std::shared_ptr<SpotLight>, ILight, PySpotLight>(io_module, "SpotLight")
    .def(py::init<
      const Vector3d&,
      const Color&,
      double,
      bool>(),
      py::arg("location"),
      py::arg("color") = Color(255, 255, 255),
      py::arg("intensity") = 1.0,
      py::arg("state") = true)
    .def_property("location",
      &SpotLight::GetLocation,
      &SpotLight::SetLocation)
    .def_property("color",
      &SpotLight::GetColor,
      &SpotLight::SetColor)
    .def_property("intensity",
      &SpotLight::GetIntensity,
      &SpotLight::SetIntensity)
    .def("fromDict", [](py::dict i_dict)
      {
      auto vec_m = py::module::import("engine.Math.Vector");
      auto color_m = py::module::import("engine.Visual");
      auto inner = i_dict["SpotLight"];
      Color color = color_m.attr("Color").attr("fromDict")(inner["Color"]).cast<Color>();
      Vector3d location = vec_m.attr("Vector3d").attr("fromDict")(inner["Location"]).cast<Vector3d>();
      double intensity = inner["Intensity"].cast<double>();
      double state = inner["State"].cast<bool>();
      return SpotLight(
        location,
        color,
        intensity,
        state);
      });
  }