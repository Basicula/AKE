import os
import sys

def add_component(module, component):
    ok = True
    include = "./"+module+"/include/"
    sources = "./"+module+"/src/"
    py = "./"+module+"/py/"
    python_tests = "./"+module+"/tests/python/"
    if not os.path.exists(include):
        ok = False
        print("Include directory missing")
    if not os.path.exists(sources):
        ok = False
        print("Sources directory missing")
    if not os.path.exists(py):
        ok = False
        print("Python module directory missing")
    if not os.path.exists(python_tests):
        ok = False
        print("Python tests directory missing")
        
    if ok:
        try:
            open(include+component+".h","x")
            open(sources+component+".cpp","x")
            open(py+component+"_py.hpp","x")
            open(python_tests+component+"_test.py","x")
            print("Successfully created")
        except:
            print("Something wrong, some files didn't created")

if __name__ == "__main__":
    add_component(sys.argv[1],sys.argv[2])