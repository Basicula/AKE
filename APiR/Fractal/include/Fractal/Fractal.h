#pragma once
#include "Macros.h"

#include <cstddef>

class Fractal
{
public:
  virtual ~Fractal() = default;

  HOSTDEVICE [[nodiscard]] virtual size_t GetValue(int i_x, int i_y) const = 0;

  void SetMaxIterations(std::size_t i_max_iterations);
  void SetScale(float i_scale);
  void SetOrigin(float i_origin_x, float i_origin_y);

protected:
  HOSTDEVICE Fractal(std::size_t i_width, std::size_t i_height, std::size_t i_max_iterations = 1000);

  HOSTDEVICE void _MapCoordinate(float& o_x, float& o_y, int i_x, int i_y) const;

  HOSTDEVICE virtual void _InitFractalRange() = 0;

protected:
  std::size_t m_width;
  std::size_t m_height;
  std::size_t m_max_iterations;
  float m_origin_x;
  float m_origin_y;
  float m_scale;

  float m_x_min;
  float m_x_max;
  float m_y_min;
  float m_y_max;
};
