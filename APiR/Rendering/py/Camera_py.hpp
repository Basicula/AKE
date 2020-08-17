#include <Rendering/Camera.h>

static void AddCamera(py::module& io_module)
  {
  py::class_<Camera>(io_module, "Camera")
    .def(py::init<
      const Vector3d&,
      const Vector3d&,
      const Vector3d&,
      double,
      double,
      double>())
    .def_property("location",
      &Camera::GetLocation,
      &Camera::SetLocation)
    .def("direction", &Camera::GetDirection)
    .def("__repr__", &Camera::Serialize)
    .def("fromDict", [](py::dict i_dict)
      {
      auto vec_m = py::module::import("engine.Math.Vector");
      auto inner = i_dict["Camera"];
      Vector3d location = vec_m.attr("Vector3d").attr("fromDict")(inner["Location"]).cast<Vector3d>();
      Vector3d lookAt = vec_m.attr("Vector3d").attr("fromDict")(inner["LookAt"]).cast<Vector3d>();
      Vector3d up = vec_m.attr("Vector3d").attr("fromDict")(inner["Up"]).cast<Vector3d>();
      double fov = inner["FoV"].cast<double>();
      double aspect = inner["Aspect"].cast<double>();
      double focusDist = inner["FocusDistance"].cast<double>();
      return Camera(
        location,
        lookAt,
        up,
        fov,
        aspect,
        focusDist);
      });
  }