#pragma once
#include <Fractal/ColorMap.h>
#include <Fractal/Fractal.h>

#include <cstdio>
#include <cstdint>

class MandelbrotSet : public Fractal
  {
  public:
    HOSTDEVICE MandelbrotSet(
      std::size_t i_width,
      std::size_t i_height,
      std::size_t i_iterations = 1000);

    HOSTDEVICE virtual Color GetColor(int i_x, int i_y) const override;

  protected:
    HOSTDEVICE virtual void _InitFractalRange() override;
  };