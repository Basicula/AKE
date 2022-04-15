#pragma once

class IShape
{
public:
  virtual ~IShape() = default;

protected:
  virtual void _CalculateBoundingBox() = 0;
};