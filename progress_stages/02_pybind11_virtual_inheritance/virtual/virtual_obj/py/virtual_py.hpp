#include <IObject.h>
#include <Object.h>

namespace py = pybind11;

namespace
{
    class PyIObject : public IObject
    {
        using IObject::IObject;
        std::string Get() const override { PYBIND11_OVERLOAD_PURE(std::string, IObject, Get,); };
    };
    
    class PyObj : public Object
    {
        using Object::Object;
        std::string Get() const override { PYBIND11_OVERLOAD(std::string, Object, Get,); };
    };
}

void AddVirtualObj(py::module& io_m)
{
    py::class_<IObject, PyIObject>(io_m, "IObject")
        .def("get", &IObject::Get);
    py::class_<Object, IObject, PyObj>(io_m, "Object")
        .def(py::init<std::string>());
}
