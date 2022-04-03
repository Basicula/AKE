#include "Fractal/Fractal.h"

Fractal::Fractal(
  std::size_t i_width,
  std::size_t i_height, 
  std::size_t i_max_iterations)
  : m_width(i_width)
  , m_height(i_height)
  , m_max_iterations(i_max_iterations)
  , m_origin_x(0.0f)
  , m_origin_y(0.0f)
  , m_scale(1.0f)
  , m_x_min(0.0f)
  , m_x_max(0.0f)
  , m_y_min(0.0f)
  , m_y_max(0.0f)
  {}

void Fractal::_MapCoordinate(float& o_x, float& o_y, int i_x, int i_y) const
  {
  o_x = ((m_x_max - m_x_min) * i_x / m_width + m_x_min) / m_scale + m_origin_x;
  o_y = ((m_y_max - m_y_min) * i_y / m_height + m_y_min) / m_scale + m_origin_y;
  }