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

#include <Scene/Scene.h>

#include <Image/Image.h>

#include <Memory/custom_vector.h>

#include <Visual/SpotLight.h>
#include <Visual/ColorMaterial.h>

#include <Rendering/RenderableObject.h>

#include <GLUTWindow/GLUTWindow.h>

#include <BMPWriter/BMPWriter.h>

#ifdef ENABLED_OPENCL
#include <OpenCLCore/OpenCLEnvironment.h>
#include <OpenCLKernels/MandelbrotSetKernel.h>
#endif

#include "CudaTest.h"

#include <iostream>
#include <chrono>

void test_bmp_writer()
  {
  Image image(800, 600, 0xff00ffff);
  BMPWriter writer;
  writer.Write("D:/Study/RayTracing/test.bmp", image);
  }

void test_fluid()
  {
  const std::size_t width = 640;
  const std::size_t height = 480;

  Scene scene("Fluid test", width, height);
  Image image(width, height, 0xffaaaaaa);

  scene.AddCamera(
    Camera(
      Vector3d(0, 5, 7),
      Vector3d(0, 0, 0),
      Vector3d(0, sqrt(2) / 2, -sqrt(2) / 2),
      75,
      width * 1.0 / height,
      2), true);
  scene.AddLight(std::make_shared<SpotLight>(Vector3d(0, 10, 0)));
  auto fluid = std::make_shared<Fluid>(48);
  scene.AddObject(fluid);

  auto update_func = [&]()
    {
    scene.RenderFrame(image);
    scene.Update();
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

void test_advanced_scene(bool i_dump_bmp = false)
  {
  const std::size_t width = 800;
  const std::size_t height = 600;

  Scene scene("Complex scene", width, height);

  auto pure_mirror = std::make_shared<ColorMaterial>(Color(0, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(1.0, 1.0, 1.0), Vector3d(1.0, 1.0, 1.0), 1, 1);
  auto more_real_mirror = std::make_shared<ColorMaterial>(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.75, 0.75, 0.75), Vector3d(1.0, 1.0, 1.0), 1, 0.75);
  auto half_mirror = std::make_shared<ColorMaterial>(Color(0, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, 0.5, 0.5), Vector3d(1.0, 1.0, 1.0), 1, 0.5);

  auto ruby = std::make_shared<ColorMaterial>(Color(255, 0, 0), Vector3d(0.1745, 0.01175, 0.01175), Vector3d(0.61424, 0.04136, 0.04136), Vector3d(0.727811, 0.626959, 0.626959), 76.8);

  auto green_plastic = std::make_shared<ColorMaterial>(Color(0, 255, 0), Vector3d(0.0, 0.05, 0.0), Vector3d(0.1, 0.35, 0.1), Vector3d(0.45, 0.55, 0.45), 32);
  auto blue_plastic = std::make_shared<ColorMaterial>(Color(0, 0, 255), Vector3d(0.0, 0.0, 0.05), Vector3d(0.1, 0.1, 0.35), Vector3d(0.45, 0.45, 0.55), 32);
  auto red_plastic = std::make_shared<ColorMaterial>(Color(255, 0, 0), Vector3d(0.05, 0.0, 0.0), Vector3d(0.5, 0.0, 0.0), Vector3d(0.7, 0.6, 0.6), 32);
  auto yellow_plastic = std::make_shared<ColorMaterial>(Color(255, 255, 0), Vector3d(0.05, 0.05, 0.0), Vector3d(0.5, 0.5, 0.0), Vector3d(0.7, 0.6, 0.6), 32);

  auto pure_glass = std::make_shared<ColorMaterial>(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), 1, 0, 1.5);

  auto water = std::make_shared<ColorMaterial>(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), 1, 0.25, 1.33);

  auto test = std::make_shared<ColorMaterial>(Color(0, 255, 0), Vector3d(0.1, 0.1, 0.1), Vector3d(0.5, 0.5, 0.5), Vector3d(0.5, 0.5, 0.5), 1);

  //double cx = -7;
  //double cy = -7;
  //for (size_t i = 0; i < 9; ++i)
  //  if (i != 4) 
  //    scene.AddObject(
  //      std::make_shared<RenderableObject>(
  //        std::make_shared<Sphere>(
  //          Vector3d(
  //            cx + 7 * (i % 3), 
  //            cy + 7 * (i / 3), 
  //            0), 
  //          2), 
  //        pure_mirror));

  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(-7, 7, 0), 3), pure_glass));
  auto sphere = std::make_shared<Sphere>(Vector3d(0, -3, 0), 0.5);
  scene.AddObject(std::make_shared<RenderableObject>(sphere, ruby));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(0, -5, 0), 2), green_plastic));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(0, 0, 0), 1), ruby));

  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(0, 10, 0), Vector3d(0, -1, 0)), blue_plastic));
  scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(0, 0, 10), Vector3d(0, 0, -1)), blue_plastic));
  scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(10, 0, 0), Vector3d(-1, 0, 0)), pure_mirror));
  scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(-10, 0, 0), Vector3d(1, 0, 0)), green_plastic));
  scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(0, -10, 0), Vector3d(0, 1, 0)), yellow_plastic));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(-5, -5, 5), Vector3d(1 / SQRT_3, 1 / SQRT_3, -1 / SQRT_3)), green_plastic));

  auto torus = std::make_shared<Torus>(Vector3d(0, 0, 0), 1, 0.5);
  scene.AddObject(std::make_shared<RenderableObject>(torus, blue_plastic));

  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Cylinder>(Vector3d(0, 0, 3), 0.49, 5), ruby));
  //std::vector<std::shared_ptr<Cylinder>> cylinders;
  //for (int i = 0; i < 25; ++i)
  //  {
  //  cylinders.emplace_back(std::make_shared<Cylinder>(Vector3d(-2.0 + i % 5, -2.0 + i / 5, 0), 0.45, 5));
  //  scene.AddObject(std::make_shared<RenderableObject>(cylinders.back(), ruby));
  //  }
  const auto cylinder = std::make_shared<Cylinder>(Vector3d(0, -2, 0), 0.5, 5);
  scene.AddObject(std::make_shared<RenderableObject>(cylinder, ruby));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Cylinder>(Vector3d(4, 4, 0), 0.5, -1), ruby));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Cylinder>(Vector3d(-4, 4, 0), 0.5, -1), ruby));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Cylinder>(Vector3d(4, -4, 0), 0.5, -1), ruby));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Cylinder>(Vector3d(-4, -4, 0), 0.5, -1), ruby));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Cylinder>(Vector3d(-27, -20, 100), 1, -1), half_mirror));

  //scene.AddLight(std::make_shared<SpotLight>(Vector3d(3, 3, -5), 0xffffffff, 0.25));
  //scene.AddLight(std::make_shared<SpotLight>(Vector3d(3, -3, -5), 0xffffffff, 0.5));
  //scene.AddLight(std::make_shared<SpotLight>(Vector3d(-3, 3, -5), 0xffffffff, 0.75));
  auto moving_light = std::make_shared<SpotLight>(Vector3d(0, 5, -5), 0xffffffff, 1);
  scene.AddLight(moving_light);
  //scene.AddLight(std::make_shared<SpotLight>(Vector3d(10, -10, -10), 0xffffffff, 1));
  //scene.AddLight(std::make_shared<SpotLight>(Vector3d(0, 0, 20), 0xffffffff, 1));

  //scene.AddCamera(Camera(Vector3d(0, 0, -10), Vector3d(0,0,0), Vector3d(0, 1, 0), 60, 1.0 * width / height, 0.5), true);
  //scene.AddCamera(Camera(Vector3d(0, 0, 0), Vector3d(-3,-3,0), Vector3d(-1/SQRT_2, 1/SQRT_2, 0), 75, 1.0 * width / height, 0.5), true);
  scene.AddCamera(Camera(Vector3d(-5, 5, -5), Vector3d(0, 0, 0), Vector3d(1 / SQRT_3, 1 / SQRT_3, 1 / SQRT_3), 75, 1.0 * width / height, 0.5), true);

  Image image(width, height);

  if (i_dump_bmp)
    {
    BMPWriter writer;
    scene.RenderFrame(image);
    writer.Write("D:\\Study\\RayTracing\\test.bmp", image);
    }

  double angle = 0.0;

  //auto sphere_scale = 1.0;
  //auto sphere_dscale = 0.1;

  //auto cylinder_scale = 1.0;
  //auto cylinder_dscale = 0.1;

  auto scene_update = [&]()
    {
    scene.RenderFrame(image);

    auto sphere_x = cos(angle - PI / 2) * 3;
    auto sphere_y = sin(angle - PI / 2) * 3;
    sphere->SetCenter(Vector3d(sphere_x, sphere_y, 0));
    //sphere->SetScale(Vector3d(1.0, 1.0, sphere_scale));
    //sphere->Rotate(Vector3d(0, 1, 0), 0.1);
    //sphere_scale += sphere_dscale;
    //if (sphere_scale >= 2.0 || sphere_scale < 1)
    //  sphere_dscale *= -1;

    //auto cylinder_x = 10.0 * std::rand() / static_cast<double>(RAND_MAX)
    //  - 5.0 * std::rand() / static_cast<double>(RAND_MAX);
    //auto cylinder_y = 10.0 * std::rand() / static_cast<double>(RAND_MAX)
    //  - 5.0 * std::rand() / static_cast<double>(RAND_MAX);
    //cylinder->SetCenter(Vector3d(cylinder_x, cylinder_y, 0.0));
    //cylinder->SetScale(Vector3d(1, 1, cylinder_scale));
    cylinder->Rotate(Vector3d(0, -1, 0), 0.1);
    //cylinder_scale += cylinder_dscale;
    //if (cylinder_scale >= 2.0 || cylinder_scale < 1.0)
    //  cylinder_dscale *= -1;
    //for (auto& cyl : cylinders)
    //  cyl->Rotate(Vector3d(0, 1, 0), 0.1);

    torus->Rotate(Vector3d(0, 1, 0), 0.1);

    const double light_x = cos(angle) * 7;
    const double light_y = sin(angle) * 7;
    moving_light->SetLocation(Vector3d(light_x, light_y, -4));
    angle += 0.1;

    scene.Update();
    };

#if false
  for (auto i = 0u; i < 30; ++i)
    scene_update();
#else
  FPSCounter fps_counter(std::cout);
  auto update_func = [&]()
    {
    scene_update();
    fps_counter.Update();
    };
  GLUTWindow window(width, height, scene.GetName().c_str());
  window.SetImageSource(&image);
  window.SetUpdateFunction(update_func);
  window.Open();
#endif
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
  //test_advanced_scene();
  //test();
  //test_opencl();
  //test_bmp_writer();
#ifdef ENABLED_CUDA
  test_cuda();
#endif
  //test_fractals();
  return 0;
  }