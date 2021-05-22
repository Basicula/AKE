#include <Main/CudaTest.h>

#include <CUDACore/KernelHandler.h>

#include <GLUTWindow/GLUTWindow.h>

#include <Fractal/MappingFunctions.h>
#include <Fractal/MandelbrotSet.h>

#include <Memory/device_ptr.h>
#include <Memory/custom_vector.h>
#include <Memory/managed_ptr.h>

#include <Image/Image.h>

#include <device_launch_parameters.h>

__global__ void mandelbrot_set_kernel(Image* iop_image, const MandelbrotSet* ip_fractal, const custom_vector<Color>* ip_colors) {
  int x = threadIdx.x;
  int y = blockIdx.x;
  iop_image->SetPixel(x, y, FractalMapping::Default(ip_fractal->GetValue(x, y), *ip_colors));
  }

void test_cuda() {
  const std::size_t width = 1024;
  const std::size_t height = 768;
  std::size_t max_iterations = 1000;
  device_ptr<MandelbrotSet> mandelbrot_set(width, height, max_iterations);
  device_vector_ptr<Color> d_colors = custom_vector<Color>::device_vector_ptr({
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
    Color(104, 51, 2) });

  KernelHandler<decltype(mandelbrot_set_kernel)> mandelbrot_set_kernel(mandelbrot_set_kernel);
  mandelbrot_set_kernel.SetNumberOfBlocks(height);
  mandelbrot_set_kernel.SetThreadsPerBlock(width);

  managed_ptr<Image> image(width, height);
  auto update_func = [&]()
    {
    mandelbrot_set_kernel.Run(image.get(), mandelbrot_set.get(), d_colors.get());
    };
  GLUTWindow window(width, height, "CudaTest");
  window.SetImageSource(image.get());
  window.SetUpdateFunction(update_func);
  window.Open();
  }

void test_cuda_rendering() {
  }