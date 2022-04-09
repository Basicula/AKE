#pragma once
#include "Geometry.2D/Shape2D.h"

#include <array>

struct Rectangle final : public Shape2D
{
  double m_width;
  double m_height;
  const std::array<Vector2d, 4> m_corners;

  Rectangle(double i_width, double i_height);

private:
  void _CalculateBoundingBox() override;
};