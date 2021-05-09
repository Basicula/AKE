#include <Fractal/MandelbrotSet.h>

MandelbrotSet::MandelbrotSet(
  std::size_t i_width,
  std::size_t i_height,
  std::size_t i_iterations)
  : Fractal(i_width, i_height, i_iterations)
  {
  _InitFractalRange();
  }

size_t MandelbrotSet::GetValue(int i_x, int i_y) const
  {
  double zx, zy, cx, cy;
  _MapCoordinate(zx, zy, i_x, i_y);
  cx = zx;
  cy = zy;
  size_t iter = 0;
  while (iter < m_max_iterations)
    {
    const double tempzx = zx * zx - zy * zy + cx;
    zy = 2 * zx * zy + cy;
    zx = tempzx;
    if (zx * zx + zy * zy > 4)
      break;
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
