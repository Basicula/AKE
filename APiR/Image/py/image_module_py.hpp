#include <Image_py.hpp>

static void AddImageSubmodule(py::module& io_module)
  {
  auto image_submodule = io_module.def_submodule("Image");
  AddImage(image_submodule);
  }