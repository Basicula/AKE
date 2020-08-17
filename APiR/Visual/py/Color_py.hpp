#include <Visual/Color.h>

namespace py = pybind11;

namespace
  {
  std::string hex_by_color_rgb(const Color& i_color)
    {
    char hex[10];
    sprintf(hex + 2, "%x", i_color.GetRGBA());
    hex[0] = '0';
    hex[1] = 'x';
    return std::string(hex);
    }
  }

static void AddColor(py::module& io_module)
  {
  py::class_<Color>(io_module, "Color")
    .def(py::init<>())
    .def(py::init<std::uint32_t>())
    .def(py::init<std::uint8_t, std::uint8_t, std::uint8_t>())
    .def(py::self * double())
    .def(py::self * Vector3d())
    .def(py::self + py::self)
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def_property("red",
                  &Color::GetRed,
                  &Color::SetRed)
    .def_property("green",
                  &Color::GetGreen,
                  &Color::SetGreen)
    .def_property("blue",
                  &Color::GetBlue,
                  &Color::SetBlue)
    .def_property("alpha",
                  &Color::GetAlpha,
                  &Color::SetAlpha)
    .def_property("rgba",
                  &Color::GetRGBA,
                  &Color::SetRGBA)
    .def("fromDict", [](py::dict i_dict)
         {
         std::uint32_t value = i_dict["Color"].cast<std::uint32_t>();
         return Color(value);
         })
    .def("__repr__", &Color::Serialize)
    .def("__hex__", hex_by_color_rgb)
    .def("__str__", hex_by_color_rgb);
  }