#pragma once
#include <Math/Vector.h>
#include <Geometry.2D/Shape.h>

struct Circle : public Shape
{
  double radius;
  Vector2d center;

  Circle(const Vector2d& i_center, double i_radius);

private:
  void _CalculateBoundingBox() override;
};
