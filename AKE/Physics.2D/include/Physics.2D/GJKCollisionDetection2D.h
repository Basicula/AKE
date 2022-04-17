#pragma once
#include "Physics.2D/Collision2D.h"
#include "Physics.2D/GJKConvex2D.h"

namespace GJKCollisionDetection2D {
  Collision2D GetCollision(const GJKConvex2D& i_first, const GJKConvex2D& i_second);
}
