#pragma once
#include <Geometry/BoundingBox.h>

class Shape
{
public:
  virtual ~Shape() = default;

  [[nodiscard]] BoundingBox2D GetBoundingBox() const;

protected:
  virtual void _CalculateBoundingBox() = 0;

protected:
  BoundingBox2D m_bounding_box;
};