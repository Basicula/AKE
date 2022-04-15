#pragma once
#include "Math/Vector.h"
#include "Physics.2D/Collider2D.h"

struct GJKConvex2D : public Collider2D
{
  [[nodiscard]] virtual Vector2d GetFurthestPoint(const Vector2d& i_direction) const = 0;
};