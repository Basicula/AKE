#pragma once
#include "Math/Vector.h"

struct Collision2D
{
  Vector2d m_normal{0.0, 0.0};
  double m_depth = 0.0;
  bool m_is_collided = false;
};