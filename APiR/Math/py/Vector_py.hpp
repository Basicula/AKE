#include "Math/Vector.h"

namespace
  {
  template<class Class>
  static void define_common(py::class_<Class>& io_class)
    {
    io_class.def(-py::self)
      .def(py::init<>())
      .def(py::init<double>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(py::self <= py::self)
      .def(py::self >= py::self)
      .def(py::self > py::self)
      .def(py::self < py::self)
      .def(-py::self)
      .def(py::self - py::self)
      .def(py::self -= py::self)
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self * float())
      .def(py::self *= float())
      .def(py::self / float())
      .def(py::self /= float())
      .def(py::self * int())
      .def(py::self *= int())
      .def(py::self / int())
      .def(py::self /= int())
      .def("dot", &Class::Dot)
      .def("normalize", &Class::Normalize)
      .def("normalized", &Class::Normalized)
      .def("length", &Class::Length)
      .def("sqrlength", &Class::SquareLength)
      .def("distance", &Class::Distance)
      .def("sqrdistance", &Class::SquareDistance)
      .def("__repr__", &Class::Serialize);
    }

  const char* g_properties[4] = { "x", "y", "z", "w" };
  template<class Class>
  static void define_individual(
    py::class_<Class>& io_class, 
    unsigned int i_num_components)
    {
    if (i_num_components < 2 || i_num_components > 4)
      return;

    auto i = 0u;
    for (; i < i_num_components; ++i)
      {
      auto get = [i](const Class& i_class)
        {
        return i_class[i];
        };
      auto set = [i](Class& i_class, double i_s)
        {
        i_class[i] = i_s;
        };
      io_class.def_property(g_properties[i], get, set);
      }
    }
  }

static void AddVectors(py::module& io_module)
  {
  auto vector_submodule = io_module.def_submodule("Vector");

  // vec2d
  using Vec2d = Vector<double, 2>;
  auto vec2d = py::class_<Vec2d>(vector_submodule, "Vector2d");
  vec2d.def(py::init<double, double>());
  vec2d.def("__str__", [](const Vec2d&) { return "Vector2d"; });
  vec2d.def("fromDict", [](py::dict i_dict)
    {
    py::list coords = i_dict["Vector2d"];
    static const std::size_t dim = py::len(coords);
    return Vec2d(
      coords[0].cast<double>(), 
      coords[1].cast<double>());
    });
  define_common(vec2d);
  define_individual(vec2d, 2);

  // vec3d
  using Vec3d = Vector<double, 3>;
  auto vec3d = py::class_<Vec3d>(vector_submodule, "Vector3d");
  vec3d.def(py::init<double, double, double>());
  vec3d.def("__str__", [](const Vec3d&) { return "Vector3d"; });
  vec3d.def("cross", &Vec3d::CrossProduct<3,double>);
  vec3d.def("fromDict", [](py::dict i_dict)
    {
    py::list coords = i_dict["Vector3d"];
    static const std::size_t dim = py::len(coords);
    return Vec3d(
      coords[0].cast<double>(), 
      coords[1].cast<double>(), 
      coords[2].cast<double>());
    });
  define_common(vec3d);
  define_individual(vec3d, 3);

  // vec4d
  using Vec4d = Vector<double, 4>;
  auto vec4d = py::class_<Vec4d>(vector_submodule, "Vector4d");
  vec4d.def(py::init<double, double, double, double>());
  vec4d.def("__str__", [](const Vec4d&) { return "Vector4d"; });
  vec4d.def("fromDict", [](py::dict i_dict)
    {
    py::list coords = i_dict["Vector4d"];
    static const std::size_t dim = py::len(coords);
    return Vec4d(
      coords[0].cast<double>(), 
      coords[1].cast<double>(), 
      coords[2].cast<double>(), 
      coords[3].cast<double>());
    });
  define_common(vec4d);
  define_individual(vec4d, 4);
  }