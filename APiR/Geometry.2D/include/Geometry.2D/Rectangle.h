#pragma once
#include "Geometry.2D/Shape.h"
#include "Math/Vector.h"

struct Rectangle : public Shape
{
  double m_width;
  double m_height;
  Vector2d m_position;

  Rectangle(const Vector2d& i_position, double i_width, double i_height);

private:
    void _CalculateBoundingBox() override;
};