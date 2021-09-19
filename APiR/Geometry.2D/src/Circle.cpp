#include <Geometry.2D/Circle.h>

Circle::Circle(const Vector2d& i_center, const double i_radius)
  : radius(i_radius)
  , center(i_center)
{
}

void Circle::_CalculateBoundingBox()
{
  m_bounding_box = BoundingBox2D(center - Vector2d(radius), center + Vector2d(radius));
}
