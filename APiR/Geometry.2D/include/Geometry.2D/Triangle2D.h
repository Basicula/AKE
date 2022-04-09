#pragma once
#include "Geometry.2D/Shape2D.h"
#include "Math/Vector.h"

#include <array>

struct Triangle2D : public Shape2D
{
  std::array<Vector2d, 3> m_vertices;

  Triangle2D(const Vector2d& i_point1, const Vector2d& i_point2, const Vector2d& i_point3);

private:
  void _CalculateBoundingBox() override;
};