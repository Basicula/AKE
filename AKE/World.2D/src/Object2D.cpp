#include "World.2D/Object2D.h"

const Shape2D& Object2D::GetShape() const
{
  return *mp_shape;
}

const SimpleDrawer& Object2D::GetDrawer() const
{
  return *mp_drawer;
}

SimpleDrawer& Object2D::GetDrawer()
{
  return *mp_drawer;
}

const GJKConvex2D* Object2D::GetCollider() const
{
  return mp_collider.get();
}

const PhysicalProperty2D* Object2D::GetPhysicalProperty() const
{
  return mp_physical_properties.get();
}

PhysicalProperty2D* Object2D::GetPhysicalProperty()
{
  return mp_physical_properties.get();
}

const Transformation2D& Object2D::GetTransformation() const
{
  return m_transformation;
}

Transformation2D& Object2D::GetTransformation()
{
  return m_transformation;
}
