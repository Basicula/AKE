#include "Image/Image.h"

static void AddImage(py::module& io_module)
  {
  py::class_<Image>(io_module, "Image")
    .def(py::init<
         std::size_t,
         std::size_t,
         std::uint32_t>(),
         py::arg("width"),
         py::arg("height"),
         py::arg("color") = 0xff000000)
    .def("getPixel", &Image::GetPixel)
    .def("setPixel", &Image::SetPixel)
    .def_property_readonly("width", &Image::GetWidth)
    .def_property_readonly("height", &Image::GetHeight)
    .def_property_readonly("size", &Image::GetSize)
    .def("data",
         [](const Image& i_image)
         {
         return std::vector<std::uint32_t>(
           i_image.GetData(), 
           i_image.GetData() + i_image.GetSize());
         })
    .def("rgbData",
         [](const Image& i_image)
         {
         return std::vector<uint8_t>(
           i_image.GetRGBAData(),
           i_image.GetRGBAData() + i_image.GetSize() * 4);
         })
     .def("rgbDataStr",
          [](const Image& i_image)
          {
          return std::wstring(
            i_image.GetRGBAData(),
            i_image.GetRGBAData() + i_image.GetSize() * 4);
          });
  }