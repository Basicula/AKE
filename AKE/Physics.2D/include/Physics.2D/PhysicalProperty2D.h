#pragma once
#include "Geometry/Transformation.h"

class PhysicalProperty2D
{
public:
  virtual ~PhysicalProperty2D() = default;

  void SetTransformationSource(Transformation2D& i_transformation);

  virtual void Apply(double i_time_delta) = 0;

protected:
  Transformation2D* mp_transformation_source = nullptr;
};