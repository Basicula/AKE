#include "Main/OpenCLExample.h"

#include "Image/Image.h"
#ifdef ENABLED_OPENCL
#include "OpenCLCore/OpenCLEnvironment.h"
#include "OpenCLKernels/MandelbrotSetKernel.h"
#endif
#include "Window/FPSCounter.h"
#include "Window/GLUTWindow.h"

#include <iostream>

void OpenCLExample()
{
#ifdef ENABLED_OPENCL
  const std::size_t width = 1024;
  const std::size_t height = 768;
  OpenCLEnvironment env;
  env.Init();
  env.PrintInfo();
  std::size_t iterations = 1000;
  MandelbrotSetKernel kernel(width, height, iterations);
  kernel.SetMaxIterations(iterations);
  kernel.SetOutput(image);
  FPSCounter fps_counter(std::cout);
  env.Build(kernel);
  const auto update_func = [&]() {
    env.Execute(kernel);
    fps_counter.Update();
  };
#if false
  for (auto i = 0; i < 30; ++i)
    update_func();
#else
  GLUTWindow window(width, height, "OpenCLTest");
  window.SetUpdateFunction(update_func);
  window.Open();
#endif
#endif
}
