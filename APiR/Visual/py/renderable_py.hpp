#include <IRenderable_py.hpp>
#include <RenderableObject_py.hpp>

static void AddRenderableSubmodule(py::module& io_module)
  {
  auto renderable_submodule = io_module.def_submodule("Renderable");
  AddIRenderable(renderable_submodule);
  AddRenderableObject(renderable_submodule);
  }