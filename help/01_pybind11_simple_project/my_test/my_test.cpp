#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>

#include <lib1_py.hpp>
#include <lib2_py.hpp>

PYBIND11_MODULE(my_test, m)
{
    AddAdderToModule(m);
    AddTestToModule(m);
}