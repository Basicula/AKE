#pragma once
#include <Macro/CudaMacro.h>

#include <Fractal/MandelbrotSet.h>
#include <Fractal/SmoothColorMap.h>

#include <Rendering/Image.h>

void cuda_fractal(
  Image* iop_image, 
  const MandelbrotSet* ip_fractal);

void test_cuda_kernel();

