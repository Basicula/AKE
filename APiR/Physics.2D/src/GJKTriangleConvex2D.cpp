#include "Physics.2D/GJKTriangleConvex2D.h"

#include "Common/Constants.h"

GJKTriangleConvex2D::GJKTriangleConvex2D(Triangle2D i_triangle)
  : m_triangle(std::move(i_triangle))
{}

GJKTriangleConvex2D::GJKTriangleConvex2D(const Vector2d& i_first, const Vector2d& i_second, const Vector2d& i_third)
  : m_triangle(i_first, i_second, i_third)
{}

Vector2d GJKTriangleConvex2D::GetFurthestPoint(const Vector2d& i_direction) const
{
  Vector2d dir = i_direction;
  if (mp_transformation_source)
    mp_transformation_source->InverseTransform(dir, true);

  Vector2d result;
  double max_dot = -MAX_DOUBLE;
  for (const auto& vertex : m_triangle.m_vertices) {
    const auto dot = vertex.Dot(dir);
    if (dot > max_dot) {
      result = vertex;
      max_dot = dot;
    }
  }

  if (mp_transformation_source)
    mp_transformation_source->Transform(result);

  return result;
}
