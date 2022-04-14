#include "Physics.2D/GJKRectangleConvex2D.h"

GJKRectangleConvex2D::GJKRectangleConvex2D(const double i_width, const double i_height)
  : m_rectangle(i_width, i_height)
{}

Vector2d GJKRectangleConvex2D::GetFurthestPoint(const Vector2d& i_direction) const
{
  Vector2d direction = i_direction;
  if (mp_transformation_source)
    mp_transformation_source->InverseTransform(direction, true);

  double max_dot = 0;
  Vector2d furthest_point;
  for (const auto& corner : m_rectangle.m_corners) {
    const auto corner_dot = corner.Dot(direction);
    if (corner_dot > max_dot) {
      max_dot = corner_dot;
      furthest_point = corner;
    }
  }

  if (mp_transformation_source)
    mp_transformation_source->Transform(furthest_point);
  return furthest_point;
}