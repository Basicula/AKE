#include "Main/FluidExample.h"

#include "Fluid/Fluid.h"
#include "Rendering/CPURayTracer.h"
#include "Rendering/Scene.h"
#include "Rendering/SimpleCamera.h"
#include "Visual/SpotLight.h"
#include "Window/GLUTWindow.h"

#include <chrono>
#include <iostream>

void FluidExample()
{
  const std::size_t width = 640;
  const std::size_t height = 480;

  Scene scene("Fluid test");
  Image image(width, height, 0xffaaaaaa);

  scene.AddCamera(
    new SimpleCamera(
      Vector3d(0, 5, 5), Vector3d(0, -1, -1), Vector3d(0, sqrt(2) / 2, -sqrt(2) / 2), 75, width * 1.0 / height),
    true);
  scene.AddLight(new SpotLight(Vector3d(0, 10, 0)));
  auto fluid = new Fluid(48);
  scene.AddObject(fluid);

  CPURayTracer renderer;
  renderer.SetOutputImage(&image);
  renderer.SetScene(&scene);
  auto update_func = [&]() {
    renderer.Render();
    fluid->Update();
  };
#ifdef _DEBUG
  for (int i = 0; i < 5; ++i) {
    std::cout << i << std::endl;
    auto start = std::chrono::system_clock::now();
    update_func();
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
  }
#else
  GLUTWindow window(width, height, "Fluid demo");
  window.SetImageSource(&image);
  window.SetUpdateFunction(update_func);
  window.Open();
#endif
}
