#include "Fractal/JuliaSet.h"

JuliaSet::JuliaSet(std::size_t i_width, std::size_t i_height, std::size_t i_iterations)
  : Fractal(i_width, i_height, i_iterations)
{
  _InitFractalRange();
  _ResetStart();
}

size_t JuliaSet::GetValue(int i_x, int i_y) const
{
  float zx, zy;
  _MapCoordinate(zx, zy, i_x, i_y);
  size_t iter = 0;
  while (iter < m_max_iterations) {
    const float tempzx = zx * zx - zy * zy + m_cx;
    zy = 2.0f * zx * zy + m_cy;
    zx = tempzx;
    if (zx * zx + zy * zy > 4.0f)
      break;
    ++iter;
  }
  return iter;
}

void JuliaSet::SetType(JuliaSetType i_type)
{
  m_type = i_type;
  _ResetStart();
}

void JuliaSet::_InitFractalRange()
{
  m_x_min = -1.8f;
  m_x_max = 1.8f;
  m_y_min = -1.5f;
  m_y_max = 1.5f;
}

void JuliaSet::_ResetStart()
{
  switch (m_type) {
    case JuliaSet::JuliaSetType::SpirallyBlob:
      m_cx = -0.11f;
      m_cy = 0.6557f;
      break;
    case JuliaSet::JuliaSetType::WhiskeryDragon:
      m_cx = -0.8f;
      m_cy = 0.15f;
      break;
    case JuliaSet::JuliaSetType::SeparatedWhorls:
      m_cx = -0.743643887037151f;
      m_cy = 0.131825904205330f;
      break;
    case JuliaSet::JuliaSetType::RomanescoBroccoli:
      m_cx = 0.0f;
      m_cy = -0.636f;
      break;
    default:
      m_cx = -0.11f;
      m_cy = 0.6557f;
      break;
  }
}
