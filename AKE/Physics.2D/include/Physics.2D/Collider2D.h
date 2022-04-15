#pragma once
#include "Geometry/Transformation.h"
#include "Physics.2D/PhysicalProperty2D.h"

struct Collider2D : public PhysicalProperty2D
{
  void SetTransformationSource(const Transformation2D* ip_transformation_source);
  
  const Transformation2D* mp_transformation_source = nullptr;
};
