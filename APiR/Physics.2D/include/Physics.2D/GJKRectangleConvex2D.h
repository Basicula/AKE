#pragma once
#include "Geometry.2D/Rectangle.h"
#include "Physics.2D/GJKConvex2D.h"

class GJKRectangleConvex2D : public GJKConvex2D
{
public:
  explicit GJKRectangleConvex2D(Rectangle i_rectangle);
  GJKRectangleConvex2D(double i_width, double i_height);

  [[nodiscard]] Vector2d GetFurthestPoint(const Vector2d& i_direction) const override;

private:
  Rectangle m_rectangle;
};