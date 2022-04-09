#pragma once
#include <Geometry.2D/Shape2D.h>

struct Circle : public Shape2D
{
  double m_radius;

  explicit Circle(double i_radius);

private:
  void _CalculateBoundingBox() override;
};
