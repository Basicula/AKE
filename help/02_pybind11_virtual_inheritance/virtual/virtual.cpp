#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>

#include <virtual_py.hpp>

PYBIND11_MODULE(virtual, m)
{
    AddVirtualObj(m);
}