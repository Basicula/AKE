#pragma once
#include "Fractal/Fractal.h"

#include <string>

class LyapunovFractal : public Fractal
  {
  public:
    LyapunovFractal(
      const std::string& i_fractal_string,
      std::size_t i_width,
      std::size_t i_height,
      std::size_t i_max_iterations = 1000);

    HOSTDEVICE virtual size_t GetValue(int i_x, int i_y) const override;

  protected:
    HOSTDEVICE virtual void _InitFractalRange();

  private:
    HOSTDEVICE double _ComputeLyapunovExponent(double i_zx, double i_zy) const;
    HOSTDEVICE double _MainFunc(std::size_t i_n, double i_zx, double i_zy) const;

  private:
    std::string m_fractal_string;
  };