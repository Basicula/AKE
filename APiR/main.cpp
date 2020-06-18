#include <iostream>
#include <chrono>

#include <Fluid.h>
#include <BoundingBox.h>
#include <Vector.h>

#include <Sphere.h>
#include <Plane.h>
#include <Cylinder.h>
#include <Torus.h>

#include <Scene.h>
#include <Image.h>
#include <SpotLight.h>
#include <GLUTWindow.h>
#include <ColorMaterial.h>
#include <BMPWriter.h>

namespace
  {

  }

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
  window.SetUpdateImageFunc(update_func);
  window.Open();
#endif
  }

void test_tree()
  {
  KDTree tree;
  tree.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(0), 10), nullptr));
  tree.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(10), 10), nullptr));
  tree.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(20), 10), nullptr));
  tree.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(30), 10), nullptr));
  tree.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(40), 10), nullptr));
  tree.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(50), 10), nullptr));
  tree.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(60), 10), nullptr));
  tree.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(70), 10), nullptr));
  }

void test_scene()
  {
  const std::size_t width = 640;
  const std::size_t height = 480;

  Image image(width, height);
  Scene scene("Test", width, height);

  scene.AddCamera(
    Camera(
      Vector3d(0, 0, -15),
      Vector3d(0),
      Vector3d(0, 1, 0),
      75,
      1024 / 768,
      0.75), true);
  scene.AddLight(std::make_shared<SpotLight>(Vector3d(0, 0, -10)));

  srand(123);
  if (false)
    for (int i = 0; i < 16; ++i)
      {
      auto x = 10.0 * std::rand() / static_cast<double>(RAND_MAX)
        - 5.0 * std::rand() / static_cast<double>(RAND_MAX);
      auto y = 10.0 * std::rand() / static_cast<double>(RAND_MAX)
        - 5.0 * std::rand() / static_cast<double>(RAND_MAX);
      auto z = 10.0 * std::rand() / static_cast<double>(RAND_MAX)
        - 5.0 * std::rand() / static_cast<double>(RAND_MAX);
      auto r = 5.0 * std::rand() / static_cast<double>(RAND_MAX);
      scene.AddObject(
        std::make_shared<RenderableObject>(
          std::make_shared<Sphere>(Vector3d(x, y, z), r),
          std::make_shared<ColorMaterial>(
            Color(255, 0, 0)
            , Vector3d(0.1745, 0.01175, 0.01175)
            , Vector3d(0.61424, 0.04136, 0.04136)
            , Vector3d(0.727811, 0.626959, 0.626959)
            , 76.8)));
      }
  else
    for (int i = 0; i < 16; ++i)
      {
      auto x = -8 + (i % 4) * 5;
      auto y = -8 + (i / 4) * 5;
      scene.AddObject(
        std::make_shared<RenderableObject>(
          std::make_shared<Sphere>(Vector3d(x, y, 0), 2),
          std::make_shared<ColorMaterial>(
            Color(255, 0, 0)
            , Vector3d(0.1745, 0.01175, 0.01175)
            , Vector3d(0.61424, 0.04136, 0.04136)
            , Vector3d(0.727811, 0.626959, 0.626959)
            , 76.8)));
      }
#ifdef _DEBUG
  const int frame_cnt = 300;
  auto start = std::chrono::system_clock::now();
  auto temp = start;
  for (int i = 0; i < frame_cnt; ++i)
    scene.RenderFrame(image);
  auto end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
    (end - start).count();
  std::cout << "Elapsed : " << elapsed <<
    ", FPS : " << 1000.0 * frame_cnt / elapsed << std::endl;
#else
  std::size_t frames = 0;
  auto start_time = std::chrono::system_clock::now();
  auto update_func = [&]()
    {
    scene.RenderFrame(image);
    scene.Update();
    ++frames;
    if (frames > 10)
      {
      auto end = std::chrono::system_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
        (end - start_time).count();
      std::cout << "FPS : " << 1000.0 * frames / elapsed << std::endl;
      frames = 0;
      start_time = end;
      }
    };
  GLUTWindow window(width, height, "Test");
  window.SetImageSource(&image);
  window.SetUpdateImageFunc(update_func);
  window.Open();
#endif
  }

void test_bmp_writer()
  {
  const std::size_t width = 800;
  const std::size_t height = 600;

  Scene scene("Complex scene", width, height);
  auto pure_mirror = std::make_shared<ColorMaterial>(Color(0, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(1.0, 1.0, 1.0), Vector3d(1.0, 1.0, 1.0), 1, 1);
  auto more_real_mirror = std::make_shared<ColorMaterial>(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.75, 0.75, 0.75), Vector3d(1.0, 1.0, 1.0), 1, 0.75);
  auto half_mirror = std::make_shared<ColorMaterial>(Color(0, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, 0.5, 0.5), Vector3d(1.0, 1.0, 1.0), 1, 0.5);
  auto ruby = std::make_shared<ColorMaterial>(Color(255, 0, 0), Vector3d(0.1745, 0.01175, 0.01175), Vector3d(0.61424, 0.04136, 0.04136), Vector3d(0.727811, 0.626959, 0.626959), 76.8);
  auto green_plastic = std::make_shared<ColorMaterial>(Color(0, 255, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.1, 0.35, 0.1), Vector3d(0.45, 0.55, 0.45), 32);
  auto blue_plastic = std::make_shared<ColorMaterial>(Color(0, 0, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.1, 0.1, 0.35), Vector3d(0.45, 0.45, 0.55), 32);
  auto red_plastic = std::make_shared<ColorMaterial>(Color(255, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, 0.0, 0.0), Vector3d(0.7, 0.6, 0.6), 32);
  auto yellow_plastic = std::make_shared<ColorMaterial>(Color(255, 255, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, 0.5, 0.0), Vector3d(0.7, 0.6, 0.6), 32);
  auto pure_glass = std::make_shared<ColorMaterial>(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), 1, 0, 1.5);
  auto water = std::make_shared<ColorMaterial>(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), 1, 0.25, 1.33);
  auto test = std::make_shared<ColorMaterial>(Color(0, 255, 0), Vector3d(0, 0, 0), Vector3d(0.5, 0.5, 0.5), Vector3d(0.5, 0.5, 0.5), 1);

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
  scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(0, 0, 0), 1), green_plastic));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(0, -5, 0), 2), green_plastic));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(-5, -5, 0), 2), green_plastic));

  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(0, 10, 0), Vector3d(0, -1, 0)), blue_plastic));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(10, 0, 0), Vector3d(-1, 0, 0)), red_plastic));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(-10, 0, 0), Vector3d(1, 0, 0)), green_plastic));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(0, -10, 0), Vector3d(0, 1, 0)), yellow_plastic));
  scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(-5, -5, 5), Vector3d(1 / SQRT_3, 1 / SQRT_3, -1 / SQRT_3)), green_plastic));

  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Torus>(Vector3d(1), 3, 1), blue_plastic));

  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Cylinder>(Vector3d(0, 0, 0), 1, 5), blue_plastic));
  scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Cylinder>(Vector3d(-3, -3, 0), 2, -1), ruby));
  //scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Cylinder>(Vector3d(-27, -20, 100), 1, -1), half_mirror));
  
  scene.AddLight(std::make_shared<SpotLight>(Vector3d(5, 5, -3), 0xffffff, 2));
  
  scene.AddCamera(Camera(Vector3d(0, 0, -5), Vector3d(0), Vector3d(0, 1, 0), 75, 1.0 * width / height, 0.5), true);

  Image image(width, height);
  
  BMPWriter writer;
  scene.RenderFrame(image);
  writer.Write("D:\\Study\\RayTracing\\test.bmp", image);

  auto scene_update = [&]()
    {
    scene.RenderFrame(image);
    scene.Update();
    };

#if false
  for (auto i = 0u; i < 10; ++i)
    scene_update();
#else
  std::size_t frames = 0;
  auto start_time = std::chrono::system_clock::now();
  auto update_func = [&]()
    {
    scene_update();
    ++frames;
    if (frames > 10)
      {
      auto end = std::chrono::system_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
        (end - start_time).count();
      std::cout << "FPS : " << 1000.0 * frames / elapsed << std::endl;
      frames = 0;
      start_time = end;
      }
    };
  GLUTWindow window(width, height, "Test");
  window.SetImageSource(&image);
  window.SetUpdateImageFunc(update_func);
  window.Open();
#endif
  }

void test()
  {
  Sphere sphere(Vector3d(0, -5, 0), 2);
  Plane plane(Vector3d(0, -10, 0), Vector3d(0, 1, 0));
  IntersectionRecord hit;
  Ray ray(Vector3d(0, 0, -5), Vector3d(2, -10, 2) - Vector3d(0,0,-5));
  plane.IntersectWithRay(hit, ray);
  Ray to_light(hit.m_intersection, Vector3d(0, 10, 0) - hit.m_intersection);
  sphere.IntersectWithRay(hit, to_light);
  }

int main()
  {
  //test_fluid();
  //test_tree();
  //test_scene();
  test_bmp_writer();
  //test();
  return 0;
  }