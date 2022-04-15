#pragma once
#include "Physics.2D/GJKConvex2D.h"

namespace GJKCollisionDetection2D {
  bool GetCollision(const GJKConvex2D& i_first, const GJKConvex2D& i_second);
}
