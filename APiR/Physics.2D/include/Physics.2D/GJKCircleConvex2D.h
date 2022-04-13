#pragma once
#include "Physics.2D/GJKConvex2D.h"

struct GJKCircleConvex2D : public GJKConvex2D
{
  double m_radius;

  explicit GJKCircleConvex2D(double i_radius);

  [[nodiscard]] Vector2d GetFurthestPoint(const Vector2d& i_direction) const override;
};
