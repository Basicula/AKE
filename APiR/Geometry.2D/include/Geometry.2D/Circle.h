#pragma once
#include <Geometry.2D/Shape.h>

struct Circle : public Shape
{
  double m_radius;

  explicit Circle(double i_radius);

private:
  void _CalculateBoundingBox() override;
};
