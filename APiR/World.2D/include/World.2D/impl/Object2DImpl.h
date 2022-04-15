#include "..\Object2D.h"
#pragma once

template <class ShapeType, class... Args>
void Object2D::InitShape(Args&&... i_args)
{
  mp_shape = std::make_unique<ShapeType>(std::forward<Args>(i_args)...);
}

template <class DrawerType, class... Args>
void Object2D::InitDrawer(Args&&... i_args)
{
  mp_drawer = std::make_unique<DrawerType>(std::forward<Args>(i_args)...);
  mp_drawer->SetTransformationSource(m_transformation);
}

template <class ColliderType, class... Args>
void Object2D::InitCollider(Args&&... i_args)
{
  mp_collider = std::make_unique<ColliderType>(std::forward<Args>(i_args)...);
  mp_collider->SetTransformationSource(&m_transformation);
}
