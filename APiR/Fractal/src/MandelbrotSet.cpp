#include "Fractal/MandelbrotSet.h"

MandelbrotSet::MandelbrotSet(const std::size_t i_width, const std::size_t i_height, const std::size_t i_iterations)
  : Fractal(i_width, i_height, i_iterations)
{
  _InitFractalRange();
}

size_t MandelbrotSet::GetValue(const int i_x, const int i_y) const
{
  float zx, zy;
  _MapCoordinate(zx, zy, i_x, i_y);
  float zx2 = zx * zx;
  float zy2 = zy * zy;
  const float cx = zx;
  const float cy = zy;
  size_t iter = 0;
  while (iter < m_max_iterations && zx2 + zy2 < 4.0f) {
    zy = 2.0f * zx * zy + cy;
    zx = zx2 - zy2 + cx;
    zx2 = zx * zx;
    zy2 = zy * zy;
    ++iter;
  }
  return iter;
}

void MandelbrotSet::_InitFractalRange()
{
  m_x_min = -2.0;
  m_x_max = 1.0;
  m_y_min = -1.0;
  m_y_max = 1.0;
}
