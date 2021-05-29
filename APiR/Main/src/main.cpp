#include <Common/ThreadPool.h>

#ifdef ENABLED_CUDA
#include <CUDACore/CUDAUtils.h>
#include <CUDACore/KernelHandler.h>

#include <Memory/device_ptr.h>
#include <Memory/managed_ptr.h>
#endif

#include <Math/Constants.h>
#include <Math/Vector.h>

#include <Fluid/Fluid.h>

#include <Fractal/MandelbrotSet.h>
#include <Fractal/JuliaSet.h>
#include <Fractal/LyapunovFractal.h>
#include <Fractal/MappingFunctions.h>

#include <Geometry/BoundingBox.h>
#include <Geometry/Sphere.h>
#include <Geometry/Plane.h>
#include <Geometry/Cylinder.h>
#include <Geometry/Torus.h>

#include <Image/Image.h>

#include <Main/SceneExamples.h>

#include <Memory/custom_vector.h>

#include <Visual/SpotLight.h>
#include <Visual/PhongMaterial.h>

#include <Rendering/RenderableObject.h>
#include <Rendering/Scene.h>
#include <Rendering/CPURayTracer.h>

#include <GLUTWindow/GLUTWindow.h>

#include <BMPWriter/BMPWriter.h>

#ifdef ENABLED_OPENCL
#include <OpenCLCore/OpenCLEnvironment.h>
#include <OpenCLKernels/MandelbrotSetKernel.h>
#endif

#include <Main/CudaTest.h>

#include <iostream>
#include <chrono>

void test_fluid()
  {
  const std::size_t width = 640;
  const std::size_t height = 480;

  Scene scene("Fluid test");
  Image image(width, height, 0xffaaaaaa);

  scene.AddCamera(
    Camera(
      Vector3d(0, 5, 7),
      Vector3d(0, 0, 0),
      Vector3d(0, sqrt(2) / 2, -sqrt(2) / 2),
      75,
      width * 1.0 / height,
      2), true);
  scene.AddLight(new SpotLight(Vector3d(0, 10, 0)));
  auto fluid = new Fluid(48);
  scene.AddObject(fluid);

  CPURayTracer renderer;
  renderer.SetOutputImage(&image);
  renderer.SetScene(&scene);
  auto update_func = [&]()
    {
    renderer.Render();
    fluid->Update();
    };
#ifdef _DEBUG
  for (int i = 0; i < 5; ++i)
    {
    std::cout << i << std::endl;
    auto start = std::chrono::system_clock::now();
    update_func();
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>
      (end - start).count() << std::endl;
    }
#else
  GLUTWindow window(width, height, "Fluid demo");
  window.SetImageSource(&image);
  window.SetUpdateFunction(update_func);
  window.Open();
#endif
  }

void test_scene()
  {
  const std::size_t width = 800;
  const std::size_t height = 600;

  Scene scene = ExampleScene::InfinityMirror();

  Image image(width, height);
  CPURayTracer renderer;
  renderer.SetOutputImage(&image);
  renderer.SetScene(&scene);

  auto update_func = [&]()
    {
    renderer.Render();
    };
  GLUTWindow window(width, height, scene.GetName().c_str());
  window.SetImageSource(&image);
  window.SetUpdateFunction(update_func);
  window.Open();
  }

#ifdef ENABLED_OPENCL
void test_opencl()
  {
  const std::size_t width = 1024;
  const std::size_t height = 768;
  Image image(width, height);
  OpenCLEnvironment env;
  env.Init();
  env.PrintInfo();
  std::size_t iterations = 1000;
  MandelbrotSetKernel kernel(width, height, iterations);
  kernel.SetMaxIterations(iterations);
  kernel.SetOutput(image);
  FPSCounter fps_counter(std::cout);
  env.Build(kernel);
  const auto update_func = [&]()
    {
    env.Execute(kernel);
    fps_counter.Update();
    };
#if false
  for (auto i = 0; i < 30; ++i)
    update_func();
#else
  GLUTWindow window(width, height, "OpenCLTest");
  window.SetImageSource(&image);
  window.SetUpdateFunction(update_func);
  window.Open();
#endif
  }
#endif

void test_fractals()
  {
  const std::size_t width = 1024;
  const std::size_t height = 768;
  Image image(width, height);
  std::size_t max_iterations = 100;
  MandelbrotSet mandelbrot_set(width, height, max_iterations);
  //JuliaSet julia_set(width, height, max_iterations);
  custom_vector<Color> color_map
    {
    Color(0, 0, 0),
    Color(66, 45, 15),
    Color(25, 7, 25),
    Color(10, 0, 45),
    Color(5, 5, 73),
    Color(0, 7, 99),
    Color(12, 43, 137),
    Color(22, 81, 175),
    Color(56, 124, 209),
    Color(132, 181, 229),
    Color(209, 234, 247),
    Color(239, 232, 191),
    Color(247, 201, 94),
    Color(255, 170, 0),
    Color(204, 127, 0),
    Color(153, 86, 0),
    Color(104, 51, 2),
    };
  //std::vector<Color> color_map
  //  {
  //  Color(0, 0, 0),
  //  Color(0, 0, 255),
  //  Color(0, 255, 0),
  //  Color(255, 0, 0),
  //  };
  //julia_set.SetColorMap(std::make_unique<SmoothColorMap>(color_map));
  //julia_set.SetType(JuliaSet::JuliaSetType::WhiskeryDragon);
  const float delta = 0.1f;
  float origin_x = 0.0f, origin_y = 0.0f;
  float scale = 1.0f;
  auto keyboard_function = [&](unsigned char i_key, int /*i_x*/, int /*i_y*/)
    {
    switch (i_key)
      {
      case 'w':
        origin_y += delta / scale;
        break;
      case 's':
        origin_y -= delta / scale;
        break;
      case 'd':
        origin_x += delta / scale;
        break;
      case 'a':
        origin_x -= delta / scale;
        break;
      case 'q':
        scale *= 1.5;
        break;
      case 'e':
        scale /= 1.5;
        break;
      case 'f':
        max_iterations += 10;
        break;
      default:
        break;
      }
    };
  auto update_func = [&]()
    {
    mandelbrot_set.SetOrigin(origin_x, origin_y);
    mandelbrot_set.SetScale(scale);
    mandelbrot_set.SetMaxIterations(max_iterations);
    //julia_set.SetOrigin(origin_x, origin_y);
    //julia_set.SetScale(scale);
    //julia_set.SetMaxIterations(max_iterations);
    ThreadPool::GetInstance()->ParallelFor(
      static_cast<std::size_t>(0),
      width * height,
      [&](std::size_t i)
      {
      int x = static_cast<int>(i % width);
      int y = static_cast<int>(i / width);
      image.SetPixel(x, y, FractalMapping::Default(mandelbrot_set.GetValue(x, y), color_map));
      //image.SetPixel(x, y, julia_set.GetColor(x, y));
      });
    };
  GLUTWindow window(width, height, "OpenCLTest");
  window.SetImageSource(&image);
  window.SetUpdateFunction(update_func);
  window.SetKeyBoardFunction(keyboard_function);
  window.Open();
  }

int main()
  {
  //test_fluid();
  test_scene();
  //test();
  //test_opencl();
#ifdef ENABLED_CUDA
  //test_cuda();
#endif
  //test_fractals();
  return 0;
  }