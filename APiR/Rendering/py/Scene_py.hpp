#include <memory>

#include <Rendering/Scene.h>

static void AddScene(py::module& io_module)
  {
  py::class_<Scene, std::shared_ptr<Scene>>(io_module, "Scene")
    .def(py::init<const std::string&>(),
      py::arg("name") = "unnamed")
    .def_property("name",
      &Scene::GetName,
      &Scene::SetName)
    .def_property("activeCamera",
      &Scene::GetActiveCameraId,
      &Scene::SetActiveCamera)
    .def_property_readonly("objCnt", &Scene::GetNumObjects)
    .def_property_readonly("camCnt", &Scene::GetNumCameras)
    .def_property_readonly("lightCnt", &Scene::GetNumLights)
    .def("addObject", &Scene::AddObject)
    .def("addCamera", 
      &Scene::AddCamera,
      py::arg("camera"),
      py::arg("setActive") = false)
    .def("addLight", &Scene::AddLight)
    .def("clearObjects", &Scene::ClearObjects)
    .def("clearCameras", &Scene::ClearCameras)
    .def("clearLights", &Scene::ClearLights)
    .def("setOnOffLight", &Scene::SetOnOffLight)
    .def("update", &Scene::Update)
    .def("__repr__", &Scene::Serialize)
    .def("fromDict", [](py::dict i_dict)
      {
      auto scene = i_dict["Scene"];
      auto name = scene["Name"].cast<std::string>();
        
      Scene res(name);

      if (scene.contains("Objects"))
        {
        auto objects = scene["Objects"];
        auto renderable_m = py::module::import("engine.Rendering");
        for (auto object : objects)
          res.AddObject(renderable_m
                        .attr("IRenderable")
                        .attr("fromDict")(object)
                        .cast<std::shared_ptr<IRenderable>>());
        }

      if (scene.contains("Lights"))
        {
        auto light_m = py::module::import("engine.Visual.Light");
        auto lights = scene["Lights"];
        for (auto light : lights)
          res.AddLight(light_m
                       .attr("ILight")
                       .attr("fromDict")(light)
                       .cast<std::shared_ptr<ILight>>());
        }

      if (scene.contains("Cameras"))
        {
        auto visual_m = py::module::import("engine.Rendering");
        auto cameras = scene["Cameras"];
        for (auto camera : cameras)
          res.AddCamera(visual_m
                        .attr("Camera")
                        .attr("fromDict")(camera)
                        .cast<Camera>());
        }

      auto active_camera = scene["ActiveCamera"].cast<std::size_t>();
      res.SetActiveCamera(active_camera);

      return res;
      });
  }