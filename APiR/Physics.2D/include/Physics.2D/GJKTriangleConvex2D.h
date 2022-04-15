#pragma once
#include "Geometry.2D/Triangle2D.h"
#include "Physics.2D/GJKConvex2D.h"

class GJKTriangleConvex2D : public GJKConvex2D
{
public:
  explicit GJKTriangleConvex2D(Triangle2D i_triangle);
  GJKTriangleConvex2D(const Vector2d& i_first, const Vector2d& i_second, const Vector2d& i_third);

  [[nodiscard]] Vector2d GetFurthestPoint(const Vector2d& i_direction) const override;

private:
  Triangle2D m_triangle;
};