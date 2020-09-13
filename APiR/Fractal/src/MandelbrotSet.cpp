#include <Fractal/MandelbrotSet.h>

MandelbrotSet::MandelbrotSet(
  std::size_t i_width,
  std::size_t i_height,
  std::size_t i_iterations)
  : Fractal(i_width, i_height, i_iterations)
  {
  _InitFractalRange();
  }

Color MandelbrotSet::GetColor(int i_x, int i_y) const
  {
  auto [zx, zy] = _MapCoordinate(i_x, i_y);
  auto [cx, cy] = std::pair<double, double>{zx, zy};
  int iter = 0;
  while (iter < m_max_iterations)
    {
    const double tempzx = zx * zx - zy * zy + cx;
    zy = 2 * zx * zy + cy;
    zx = tempzx;
    if (zx * zx + zy * zy > 4)
      break;
    ++iter;
    }
  return (*m_color_map)(iter, m_max_iterations);
  }

void MandelbrotSet::_InitFractalRange()
  {
  m_x_min = -2.0;
  m_x_max = 1.0;
  m_y_min = -1.0;
  m_y_max = 1.0;
  }
