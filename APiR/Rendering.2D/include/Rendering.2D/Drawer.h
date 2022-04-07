#pragma once
#include "Geometry/Transformation.h"

class Drawer
{
public:
  virtual ~Drawer() = default;

  virtual void Draw() const = 0;

  void SetTransformationSource(const Transformation2D& i_transformation);

protected:
  const Transformation2D* mp_transformation = nullptr;
};