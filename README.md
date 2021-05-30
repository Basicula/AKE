[![Badge](https://img.shields.io/badge/C++-birghtgreen)](https://github.com/Basicula)
[![Badge](https://img.shields.io/badge/C-birghtgreen)](https://github.com/Basicula)
[![Badge](https://img.shields.io/badge/Python-blue)](https://github.com/Basicula)
[![Badge](https://img.shields.io/badge/CMake-aaaaaa)](https://github.com/Basicula)
[![Badge](https://europe-west6-xlocc-badge.cloudfunctions.net/XLOCC/Basicula/APiR?kill_cache=7&ifiles=pybind11|ThirdParties)](https://github.com/Basicula)

# **APiR** - **A**ll **P**ossible **i**n **R**endering

## Actual progress
![](States/actual_state.png)

## Instalation
Next thirdparties must be installed before configuring cmake
or they will be taken from ThirdParties folder instead
1. OpenCL
2. GLUT
3. CUDA

## Feature list
- [ ] Geometry
    - [ ] Primitives
        - [x] Sphere
        - [x] Cylinder
        - [x] Plane
        - [x] Torus
        - [ ] Cone
        - [ ] Cube
    - [ ] Mesh
- [ ] Fractal
    - [ ] 2D
        - [x] Mandelbrot set
        - [x] Julia set
    - [ ] 3D
- [ ] Rendering worflow
    - [ ] RayTracing
        - [x] CPU version
        - [ ] CUDA version
    - [ ] RayMarching
    - [ ] PathTracing
- [ ] Physic
- [ ] Serialization
- [ ] Python wrapper
- [ ] Camera controller
- [ ] Memory
    - [x] Custom vector
    - [x] CUDA managed memory pointer/allocator
    - [x] CUDA device memory pointer/allocator
    - [x] Memory manager
- [ ] Lighting
    - [x] Spot light
    - [ ] Direction light
    - [ ] Object light emmiter
- [x] Window
    - [x] Events
        - [x] Mouse
            - [x] Button pressed
            - [x] Button released
            - [x] Scroll
        - [x] Keyboard
            - [x] Key pressed
            - [x] Key repeat
            - [x] Key released
        - [x] Window
            - [x] Window close
            - [x] Window resize
    - [x] GLUT window
    - [x] GLFW window
