#include <ILight.h>

namespace
  {
  class PyILight : public ILight
    {
    using ILight::ILight;
    void SetState(bool i_state) override
      {
      PYBIND11_OVERLOAD_PURE(
        void,
        ILight,
        SetState,
        i_state);
      }
    bool GetState() const override
      {
      PYBIND11_OVERLOAD_PURE(
        bool,
        ILight,
        GetState,
        );
      }
    std::string Serialize() const override
      {
      PYBIND11_OVERLOAD_PURE(
        std::string,
        ILight,
        Serialize,
        );
      }
    Vector3d GetDirection(const Vector3d& i_point) const override
      {
      PYBIND11_OVERLOAD_PURE(
        Vector3d,
        ILight,
        GetDirection,
        i_point
        );
      }
    };
  }

static void AddILight(py::module& io_module)
  {
  py::class_<ILight, std::shared_ptr<ILight>, PyILight>(io_module, "ILight")
    .def_property("state",
      &ILight::GetState,
      &ILight::SetState)
    .def("direction", &ILight::GetDirection)
    .def("__repr__", &ILight::Serialize)
    .def("fromDict", [](py::dict i_dict)
      {
      auto light_m = py::module::import("engine.Visual.Light");
      if (i_dict.contains("SpotLight"))
        return light_m.attr("SpotLight").attr("fromDict")(i_dict).cast<std::shared_ptr<ILight>>();
      return std::shared_ptr<ILight>();
      });
  }