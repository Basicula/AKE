#pragma once
#include <Geometry/BoundingBox.h>

struct Shape
{
  BoundingBox2D m_bounding_box;

  virtual ~Shape() = default;

protected:
  virtual void _CalculateBoundingBox() = 0;
};