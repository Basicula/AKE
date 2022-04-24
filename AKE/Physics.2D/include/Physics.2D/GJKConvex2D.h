#pragma once
#include "Math/Vector.h"
#include "Physics.2D/PhysicalProperty2D.h"

struct GJKConvex2D : public PhysicalProperty2D
{
  [[nodiscard]] virtual Vector2d GetFurthestPoint(const Vector2d& i_direction) const = 0;

private:
  void Apply(double) override {}
};