#include "Geometry.2D/Rectangle.h"

Rectangle::Rectangle(const Vector2d& i_position, const double i_width, const double i_height)
  : m_width(i_width)
  , m_height(i_height)
  , m_position(i_position)
{}

void Rectangle::_CalculateBoundingBox()
{
  const double half_width = m_width / 2;
  const double half_height = m_height / 2;
  m_bounding_box =
    BoundingBox2D{ m_position - Vector2d{ half_width, half_height }, m_position + Vector2d{ half_width, half_height } };
}
