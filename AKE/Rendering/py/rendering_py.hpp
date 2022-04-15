#include <IRenderable_py.hpp>
#include <RenderableObject_py.hpp>
#include <Camera_py.hpp>
#include <Scene_py.hpp>

static void AddRenderingSubmodule(py::module& io_module)
  {
  auto rendering_submodule = io_module.def_submodule("Rendering");
  AddIRenderable(rendering_submodule);
  AddRenderableObject(rendering_submodule);
  AddCamera(rendering_submodule);
  AddScene(rendering_submodule);
  }