#include "Common/ThreadPool.h"

#ifdef ENABLED_CUDA
#include "CUDACore/CUDAUtils.h"
#include "CUDACore/KernelHandler.h"

#include <Memory/device_ptr.h>
#include <Memory/managed_ptr.h>
#endif

#include "BMPWriter/BMPWriter.h"
#include "Fluid/Fluid.h"
#include "Fractal/JuliaSet.h"
#include "Fractal/LyapunovFractal.h"
#include "Fractal/MandelbrotSet.h"
#include "Fractal/MappingFunctions.h"
#include "Geometry.2D/Circle.h"
#include "Geometry.2D/Rectangle.h"
#include "Geometry.3D/Cylinder.h"
#include "Geometry.3D/Plane.h"
#include "Geometry.3D/Sphere.h"
#include "Geometry.3D/Torus.h"
#include "Geometry/BoundingBox.h"
#include "Image/Image.h"
#include "Main/ConsoleLogEventListner.h"
#include "Main/2DSceneExample.h"
#include "Main/SimpleCameraController.h"
#include "Math/Constants.h"
#include "Math/Vector.h"
#include "Rendering.2D/CircleDrawer.h"
#include "Rendering.2D/RectangleDrawer.h"
#include "Rendering.2D/OpenGLRenderer.h"
#include "Rendering.2D/Scene2D.h"
#include "Rendering.2D/Object2D.h"
#include "Rendering/CPURayTracer.h"
#include "Rendering/RenderableObject.h"
#include "Rendering/Scene.h"
#include "Rendering/SimpleCamera.h"
#include "Visual/PhongMaterial.h"
#include "Visual/SpotLight.h"
#include "Window/GLFWDebugGUIView.h"
#include "Window/GLFWWindow.h"
#include "Window/GLUTWindow.h"
#include "Window/KeyboardEvent.h"

#include <Memory/custom_vector.h>

#include "Main/CudaTest.h"

#include <chrono>
#include <iostream>

void test_event_listner()
{
  const std::size_t width = 1024;
  const std::size_t height = 768;
  // GLUTWindow window(width, height, "EventListnerTest");
  GLFWWindow window(width, height, "EventListnerTest");
  window.SetEventListner(new ConsoleLogEventListner);
  window.Open();
}

void test_gui_view()
{
  const std::size_t width = 1024;
  const std::size_t height = 768;
  GLFWWindow window(width, height, "GUIView");
  window.SetGUIView(new GLFWDebugGUIView(window.GetOpenGLWindow()));
  window.Open();
}

int main()
{
   SceneExample2D();
  // test();
  // test_opencl();
#ifdef ENABLED_CUDA
  // test_cuda();
#endif
  // test_fractals();
  // test_event_listner();
  // test_gui_view();
  return 0;
}