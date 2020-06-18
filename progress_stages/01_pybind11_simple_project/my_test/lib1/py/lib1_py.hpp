#include <lib1.h>

namespace py = pybind11;

void AddAdderToModule(py::module& io_m)
{
    auto adder_sub = io_m.def_submodule("Adder");
    
    py::class_<Adder>(adder_sub, "Adder")
        .def(py::init<>())
        .def("get", &Adder::get)
        .def("push", &Adder::push);
}
