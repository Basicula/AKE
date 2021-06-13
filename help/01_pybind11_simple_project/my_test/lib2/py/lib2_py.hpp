#include <lib2.h>

namespace py = pybind11;

void AddTestToModule(py::module& io_m)
{
    auto test_sub = io_m.def_submodule("Test");
    py::class_<Test>(test_sub, "Test")
        .def(py::init<>())
        .def("hello", &Test::hello);
}