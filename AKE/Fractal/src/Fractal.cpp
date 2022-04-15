#include "Fractal/Fractal.h"

Fractal::Fractal(const std::size_t i_width, const std::size_t i_height, const std::size_t i_max_iterations)
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

void Fractal::_MapCoordinate(float& o_x, float& o_y, const int i_x, const int i_y) const
{
  o_x = ((m_x_max - m_x_min) * i_x / m_width + m_x_min) / m_scale + m_origin_x;
  o_y = ((m_y_max - m_y_min) * i_y / m_height + m_y_min) / m_scale + m_origin_y;
}

void Fractal::SetMaxIterations(const std::size_t i_max_iterations)
{
  m_max_iterations = i_max_iterations;
}

void Fractal::SetScale(const float i_scale)
{
  m_scale = i_scale;
}

void Fractal::SetOrigin(const float i_origin_x, const float i_origin_y)
{
  m_origin_x = i_origin_x;
  m_origin_y = i_origin_y;
}
