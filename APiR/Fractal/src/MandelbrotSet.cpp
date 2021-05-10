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
  double zx2 = zx * zx;
  double zy2 = zy * zy;
  cx = zx;
  cy = zy;
  size_t iter = 0;
  while (iter < m_max_iterations && zx2 + zy2 < 4.0)
    {
    zy = 2.0 * zx * zy + cy;
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
