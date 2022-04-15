#pragma once
#include <Geometry/BoundingBox.h>
#include <Geometry/IShape.h>

struct Shape2D : public IShape
{
  BoundingBox2D m_bounding_box;
};