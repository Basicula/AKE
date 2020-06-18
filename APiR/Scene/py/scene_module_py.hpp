#include <Scene_py.hpp>

static void AddSceneSubmodule(py::module& io_module)
  {
  AddScene(io_module);
  }