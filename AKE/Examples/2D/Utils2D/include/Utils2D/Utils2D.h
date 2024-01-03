#pragma once
#include "World.2D/Object2D.h"

#include <memory>

namespace Utils2D {
  std::unique_ptr<Object2D> RandomObject();
  std::unique_ptr<Object2D> RandomCircle(double i_min_radius, double i_max_radius);
  std::unique_ptr<Object2D> RandomRectangle(double i_min_side_length, double i_max_side_length);
  std::unique_ptr<Object2D> RandomTriangle(double i_max_side_length);
  Vector2d RandomUnitVector(double i_length = 1.0);
}
