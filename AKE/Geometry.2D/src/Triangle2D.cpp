#include "Geometry.2D/Triangle2D.h"

Triangle2D::Triangle2D(const Vector2d& i_point1, const Vector2d& i_point2, const Vector2d& i_point3)
  : m_vertices{ i_point1, i_point2, i_point3 }
{}

void Triangle2D::_CalculateBoundingBox()
{
  m_bounding_box = BoundingBox2D::Make(m_vertices.begin(), m_vertices.end());
}
