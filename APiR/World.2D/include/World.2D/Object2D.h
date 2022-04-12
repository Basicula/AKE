#pragma once
#include "Geometry.2D/Shape2D.h"
#include "Geometry/Transformation.h"
#include "Rendering.2D/Drawer.h"

#include <memory>

struct Object2D
{
  std::unique_ptr<Shape2D> mp_shape;
  std::unique_ptr<Drawer> mp_drawer;
  Transformation2D m_transformation;
};
