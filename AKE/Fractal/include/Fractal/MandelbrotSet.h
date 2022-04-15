#pragma once
#include "Fractal/Fractal.h"

class MandelbrotSet final : public Fractal
{
public:
  HOSTDEVICE MandelbrotSet(std::size_t i_width, std::size_t i_height, std::size_t i_iterations = 1000);

  HOSTDEVICE [[nodiscard]] size_t GetValue(int i_x, int i_y) const override;

protected:
  HOSTDEVICE void _InitFractalRange() override;
};