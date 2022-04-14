#pragma once
#include "Geometry.2D/Rectangle.h"
#include "Physics.2D/GJKConvex2D.h"

struct GJKRectangleConvex2D : public GJKConvex2D
{
  Rectangle m_rectangle;

  GJKRectangleConvex2D(double i_width, double i_height);

  [[nodiscard]] Vector2d GetFurthestPoint(const Vector2d& i_direction) const override;
};