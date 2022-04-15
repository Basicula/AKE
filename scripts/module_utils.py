import argparse
from ast import parse
import os
import sys

def create_module(module_name, has_python, has_cpp_tests, has_py_tests):
    project_folder_root = os.path.abspath("./AKE")

    module_path = f"{project_folder_root}/{module_name}"
    if os.path.exists(module_path):
        raise f"Module with name {module_name} already exists"
        
    os.mkdir(module_path)

    include = f"{module_path}/include"
    headers = f"{include}/{module_name}"
    os.mkdir(include)
    os.mkdir(headers)
    src = f"{module_path}/src"
    os.mkdir(src)
    if has_python:
        py = f"{module_path}/py"
        os.mkdir(py)
    if has_cpp_tests or has_py_tests:
        tests = f"{module_path}/tests"
        os.mkdir(tests)
        if has_cpp_tests:
            cpp_tests = f"{tests}/c++"
            os.mkdir(cpp_tests)
        if has_py_tests:
            py_tests = f"{tests}/python"
            os.mkdir(py_tests)

    with open(f"{module_path}/CMakeLists.txt", 'w') as fo:
        fo.write("set(\n")
        fo.write("  SOURCES\n")
        fo.write(")\n\n")

        fo.write("set(\n")
        fo.write("  HEADERS\n")
        fo.write(")\n\n")

        fo.write("set(\n")
        fo.write("  LINKS\n")
        fo.write(")\n\n")

        if has_python:
            fo.write("set(\n")
            fo.write("  PY\n")
            fo.write(")\n\n")

        if has_cpp_tests:
            fo.write("set(\n")
            fo.write("  TESTS\n")
            fo.write(")\n\n")

        fo.write("generate_project(\n")
        fo.write("  STATIC\n")
        fo.write("  SOURCES ${SOURCES}\n")
        fo.write("  HEADERS ${HEADERS}\n")
        if has_python:
            fo.write("  PY      ${PY}\n")
        if has_cpp_tests:
            fo.write("  TESTS   ${TESTS}\n")
        fo.write("  LINK    ${LINKS}\n")
        fo.write(")\n")

def add_component(component, module, with_py, with_cpp_tests, with_py_tests):
    project_folder_root = os.path.abspath("./AKE/")
    module_path = f"{project_folder_root}/{module}"
    
    open(f"{module_path}/include/{module}/{component}.h", 'x')
    open(f"{module_path}/src/{component}.cpp", 'x')
    if with_py:
        open(f"{module_path}/py/{component}_py.hpp", 'x')
    if with_cpp_tests:
        open(f"{module_path}/tests/c++/{component}_test.cpp", 'x')
    if with_py_tests:
        open(f"{module_path}/tests/python/{component}_test.py", 'x')

if __name__ == "__main__":
    help_parser = argparse.ArgumentParser()

    command_parsers = []

    new_module_parser = argparse.ArgumentParser(exit_on_error=False)
    new_module_parser.add_argument("--new_module", required=True, dest="module_name")
    new_module_parser.add_argument("-py", required=False, action="store_true")
    new_module_parser.add_argument("-tcpp", "--tests_cpp", required=False, action="store_true")
    new_module_parser.add_argument("-tpy", "--tests_py", required=False, action="store_true")
    command_parsers.append([new_module_parser, create_module])

    add_component_parser = argparse.ArgumentParser(exit_on_error=False)
    add_component_parser.add_argument("--new_component", required=True, dest="component_name")
    add_component_parser.add_argument("--module", required=True, dest="module_name")
    add_component_parser.add_argument("-py", required=False, action="store_true")
    add_component_parser.add_argument("-tcpp", "--tests_cpp", required=False, action="store_true")
    add_component_parser.add_argument("-tpy", "--tests_py", required=False, action="store_true")
    command_parsers.append([add_component_parser, add_component])

    for parser_data in command_parsers:
        try:
            parser = parser_data[0]
            processor = parser_data[1]
            args = vars(parser.parse_args(sys.argv[1:]))
            arg_values = []
            for arg_name in args:
                arg_values.append(args[arg_name])
            processor(*arg_values)
            break
        except:
            continue