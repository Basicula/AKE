#include "Physics.2D/PhysicalProperty2D.h"

void PhysicalProperty2D::SetTransformationSource(Transformation2D& i_transformation)
  {
  mp_transformation_source = &i_transformation;
  }