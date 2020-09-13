#include <Fractal/JuliaSet.h>

JuliaSet::JuliaSet(
  std::size_t i_width,
  std::size_t i_height,
  std::size_t i_iterations)
  : Fractal(i_width, i_height, i_iterations)
  {
  _InitFractalRange();
  _ResetStart();
  }

Color JuliaSet::GetColor(int i_x, int i_y) const
  {
  auto [zx, zy] = _MapCoordinate(i_x, i_y);
  int iter = 0;
  while (iter < m_max_iterations)
    {
    const double tempzx = zx * zx - zy * zy + m_cx;
    zy = 2 * zx * zy + m_cy;
    zx = tempzx;
    if (zx * zx + zy * zy > 4)
      break;
    ++iter;
    }
  return (*m_color_map)(iter, m_max_iterations);
  }

void JuliaSet::SetType(JuliaSetType i_type)
  {
  m_type = i_type;
  _ResetStart();
  }

void JuliaSet::_InitFractalRange()
  {
  m_x_min = -1.8;
  m_x_max = 1.8;
  m_y_min = -1.5;
  m_y_max = 1.5;
  }

void JuliaSet::_ResetStart()
  {
  switch (m_type)
    {
    case JuliaSet::JuliaSetType::SpirallyBlob:
      m_cx = -0.11;
      m_cy = 0.6557;
      break;
    case JuliaSet::JuliaSetType::WhiskeryDragon:
      m_cx = -0.8;  
      m_cy = 0.15;
      break;
    case JuliaSet::JuliaSetType::SeparatedWhorls:
      m_cx = -0.743643887037151l; 
      m_cy = 0.131825904205330;
      break;
    case JuliaSet::JuliaSetType::RomanescoBroccoli:
      m_cx = 0.0; 
      m_cy = -0.636;
      break;
    default:
      m_cx = -0.11;
      m_cy = 0.6557;
      break;
    }
  }
