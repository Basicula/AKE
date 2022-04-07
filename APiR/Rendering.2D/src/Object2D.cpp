#include "Rendering.2D/Object2D.h"

const Shape& Object2D::GetShape() const
{
  return *mp_shape;
}

const Drawer& Object2D::GetDrawer() const
{
  return *mp_drawer;
}

const Transformation2D& Object2D::GetTransformation() const
{
  return m_transformation;
}

Transformation2D& Object2D::GetTransformation()
{
  return m_transformation;
}
