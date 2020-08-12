#include <iostream>
#include <chrono>

#include <Common/BoundingBox.h>
#include <Common/DefinesAndConstants.h>

#include <Math/Vector.h>

#include <Fluid/Fluid.h>

#include <Primitives/Sphere.h>
#include <Primitives/Plane.h>
#include <Primitives/Cylinder.h>
#include <Primitives/Torus.h>

#include <Scene.h>
#include <Image.h>
#include <SpotLight.h>
#include <ColorMaterial.h>

#include <GLUTWindow/GLUTWindow.h>

#include <BMPWriter/BMPWriter.h>

#include <OpenCLCore/OpenCLEnvironment.h>
#include <OpenCLCore/MandelbrotSetKernel.h>

void test_fluid()
  {
  const std::size_t width = 640;
  const std::size_t height = 480;

  Scene scene("Fluid test", width, height);
  Image image(width, height, Color(0xaaaaaa));

  scene.AddCamera(
    Camera(
      Vector3d(0, 5, 7),
      Vector3d(0, 0, 0),
      Vector3d(0, sqrt(2)/2, -sqrt(2)/2),
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

  auto pure_mirror      = std::make_shared<ColorMaterial>(Color(0, 0, 0),       Vector3d(0.0, 0.0, 0.0), Vector3d(1.0, 1.0, 1.0),     Vector3d(1.0, 1.0, 1.0), 1, 1);
  auto more_real_mirror = std::make_shared<ColorMaterial>(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.75, 0.75, 0.75),  Vector3d(1.0, 1.0, 1.0), 1, 0.75);
  auto half_mirror      = std::make_shared<ColorMaterial>(Color(0, 0, 0),       Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, 0.5, 0.5),     Vector3d(1.0, 1.0, 1.0), 1, 0.5);

  auto ruby = std::make_shared<ColorMaterial>(Color(255, 0, 0), Vector3d(0.1745, 0.01175, 0.01175), Vector3d(0.61424, 0.04136, 0.04136), Vector3d(0.727811, 0.626959, 0.626959), 76.8);

  auto green_plastic  = std::make_shared<ColorMaterial>(Color(0, 255, 0),   Vector3d(0.0, 0.05, 0.0), Vector3d(0.1, 0.35, 0.1),  Vector3d(0.45, 0.55, 0.45), 32);
  auto blue_plastic   = std::make_shared<ColorMaterial>(Color(0, 0, 255),   Vector3d(0.0, 0.0, 0.05), Vector3d(0.1, 0.1, 0.35),  Vector3d(0.45, 0.45, 0.55), 32);
  auto red_plastic    = std::make_shared<ColorMaterial>(Color(255, 0, 0),   Vector3d(0.05, 0.0, 0.0), Vector3d(0.5, 0.0, 0.0),   Vector3d(0.7, 0.6, 0.6),    32);
  auto yellow_plastic = std::make_shared<ColorMaterial>(Color(255, 255, 0), Vector3d(0.05, 0.05, 0.0), Vector3d(0.5, 0.5, 0.0),   Vector3d(0.7, 0.6, 0.6),    32);

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
  
  //scene.AddLight(std::make_shared<SpotLight>(Vector3d(3, 3, -5), 0xffffff, 0.25));
  //scene.AddLight(std::make_shared<SpotLight>(Vector3d(3, -3, -5), 0xffffff, 0.5));
  //scene.AddLight(std::make_shared<SpotLight>(Vector3d(-3, 3, -5), 0xffffff, 0.75));
  auto moving_light = std::make_shared<SpotLight>(Vector3d(0, 5, -5), 0xffffff, 1);
  scene.AddLight(moving_light);
  //scene.AddLight(std::make_shared<SpotLight>(Vector3d(10, -10, -10), 0xffffff, 1));
  //scene.AddLight(std::make_shared<SpotLight>(Vector3d(0, 0, 20), 0xffffff, 1));
  
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
    sphere->SetCenter(Vector3d(sphere_x,sphere_y,0));
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

void test()
  {
  //Sphere sphere(Vector3d(0, -5, 0), 2);
  //Plane plane(Vector3d(0, -10, 0), Vector3d(0, 1, 0));
  //IntersectionRecord hit;
  //Ray ray(Vector3d(0, 0, -5), Vector3d(2, -10, 2) - Vector3d(0,0,-5));
  //plane.IntersectWithRay(hit, ray);
  //Ray to_light(hit.m_intersection, Vector3d(0, 10, 0) - hit.m_intersection);
  //sphere.IntersectWithRay(hit, to_light);

  Cylinder cylinder(Vector3d(0, 0, 0), 2, 2);
  IntersectionRecord hit;
  Ray ray(Vector3d(0,0,-3), Vector3d(0, 0, 1));
  cylinder.IntersectWithRay(hit, ray);
  }

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

int main()
  {
  //test_fluid();
  //test_advanced_scene();
  //test();
  test_opencl();
  return 0;
  }