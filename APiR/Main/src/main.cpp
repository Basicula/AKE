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

void test_fractals()
{
  const std::size_t width = 1024;
  const std::size_t height = 768;
  Image image(width, height);
  std::size_t max_iterations = 100;
  // std::unique_ptr<Fractal> p_fractal = std::make_unique<MandelbrotSet>(width, height, max_iterations);
  std::unique_ptr<Fractal> p_fractal = std::make_unique<JuliaSet>(width, height, max_iterations);
  custom_vector<Color> color_map{
    Color(0, 0, 0),       Color(66, 45, 15),    Color(25, 7, 25),    Color(10, 0, 45),    Color(5, 5, 73),
    Color(0, 7, 99),      Color(12, 43, 137),   Color(22, 81, 175),  Color(56, 124, 209), Color(132, 181, 229),
    Color(209, 234, 247), Color(239, 232, 191), Color(247, 201, 94), Color(255, 170, 0),  Color(204, 127, 0),
    Color(153, 86, 0),    Color(104, 51, 2),
  };
  // std::vector<Color> color_map
  //  {
  //  Color(0, 0, 0),
  //  Color(0, 0, 255),
  //  Color(0, 255, 0),
  //  Color(255, 0, 0),
  //  };
  // julia_set.SetColorMap(std::make_unique<SmoothColorMap>(color_map));
  // julia_set.SetType(JuliaSet::JuliaSetType::WhiskeryDragon);
  class FractalChangeEventListner : public EventListner
  {
  public:
    FractalChangeEventListner(Fractal* ip_fractal)
      : mp_fractal(ip_fractal)
    {}

    virtual void PollEvents() override{};

  protected:
    virtual void _ProcessEvent(const Event& i_event) override
    {
      if (i_event.Type() != Event::EventType::KEY_PRESSED_EVENT)
        return;
      const auto& key_pressed_event = static_cast<const KeyPressedEvent&>(i_event);
      switch (key_pressed_event.Key()) {
        case KeyboardButton::KEY_W:
          origin_y += delta / scale;
          break;
        case KeyboardButton::KEY_S:
          origin_y -= delta / scale;
          break;
        case KeyboardButton::KEY_D:
          origin_x += delta / scale;
          break;
        case KeyboardButton::KEY_A:
          origin_x -= delta / scale;
          break;
        case KeyboardButton::KEY_Q:
          scale *= 1.5;
          break;
        case KeyboardButton::KEY_E:
          scale /= 1.5;
          break;
        default:
          break;
      }
      mp_fractal->SetOrigin(origin_x, origin_y);
      mp_fractal->SetScale(scale);
    }

  private:
    Fractal* mp_fractal;
    const float delta = 0.1f;
    float origin_x = 0.0f, origin_y = 0.0f;
    float scale = 1.0f;
  } event_listner(p_fractal.get());
  auto update_func = [&]() {
    Parallel::ThreadPool::GetInstance()->ParallelFor(static_cast<std::size_t>(0), width * height, [&](std::size_t i) {
      int x = static_cast<int>(i % width);
      int y = static_cast<int>(i / width);
      image.SetPixel(x, y, static_cast<std::uint32_t>(FractalMapping::Default(p_fractal->GetValue(x, y), color_map)));
      // image.SetPixel(x, y, julia_set.GetColor(x, y));
    });
  };
  GLUTWindow window(width, height, "FracralsTest");
  window.SetImageSource(&image);
  window.SetUpdateFunction(update_func);
  window.SetEventListner(&event_listner);
  window.Open();
}

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