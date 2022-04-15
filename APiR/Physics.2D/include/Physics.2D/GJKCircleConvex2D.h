#pragma once
#include "Geometry.2D/Circle.h"
#include "Physics.2D/GJKConvex2D.h"

class GJKCircleConvex2D : public GJKConvex2D
{
public:
  explicit GJKCircleConvex2D(Circle i_circle);
  explicit GJKCircleConvex2D(double i_radius);

  [[nodiscard]] Vector2d GetFurthestPoint(const Vector2d& i_direction) const override;

private:
  Circle m_circle;
};
