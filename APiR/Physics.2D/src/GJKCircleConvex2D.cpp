#include "Physics.2D/GJKCircleConvex2D.h"

GJKCircleConvex2D::GJKCircleConvex2D(const double i_radius)
  : m_radius(i_radius)
{}

Vector2d GJKCircleConvex2D::GetFurthestPoint(const Vector2d& i_direction) const
{
  Vector2d result(0.0);
  if (mp_transformation_source)
    result = mp_transformation_source->GetTranslation();
  result += i_direction.Normalized() * m_radius;
  return result;
}
