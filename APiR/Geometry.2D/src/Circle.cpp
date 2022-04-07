#include <Geometry.2D/Circle.h>

Circle::Circle(const double i_radius)
  : m_radius(i_radius)
{}

void Circle::_CalculateBoundingBox()
{
  m_bounding_box = BoundingBox2D(Vector2d{ -m_radius, -m_radius }, Vector2d{ m_radius, m_radius });
}
