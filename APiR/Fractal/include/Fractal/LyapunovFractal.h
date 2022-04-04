#pragma once
#include "Fractal/Fractal.h"

#include <string>

class LyapunovFractal final : public Fractal
{
public:
  LyapunovFractal(std::string i_fractal_string,
                  std::size_t i_width,
                  std::size_t i_height,
                  std::size_t i_max_iterations = 1000);

  HOSTDEVICE [[nodiscard]] size_t GetValue(int i_x, int i_y) const override;

protected:
  HOSTDEVICE void _InitFractalRange() override;

private:
  HOSTDEVICE [[nodiscard]] float _ComputeLyapunovExponent(float i_zx, float i_zy) const;
  HOSTDEVICE [[nodiscard]] float _MainFunc(std::size_t i_n, float i_zx, float i_zy) const;

private:
  std::string m_fractal_string;
};