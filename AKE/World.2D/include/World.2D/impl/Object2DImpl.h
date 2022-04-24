#pragma once

template <class ShapeType, class... Args, std::enable_if_t<std::is_base_of_v<Shape2D, ShapeType>, bool>>
void Object2D::InitShape(Args&&... i_args)
{
  mp_shape = std::make_unique<ShapeType>(std::forward<Args>(i_args)...);
}

template <class DrawerType, class... Args, std::enable_if_t<std::is_base_of_v<SimpleDrawer, DrawerType>, bool>>
void Object2D::InitDrawer(Args&&... i_args)
{
  mp_drawer = std::make_unique<DrawerType>(std::forward<Args>(i_args)...);
  mp_drawer->SetTransformationSource(m_transformation);
}

template <class ColliderType, class... Args, std::enable_if_t<std::is_base_of_v<GJKConvex2D, ColliderType>, bool>>
void Object2D::InitCollider(Args&&... i_args)
{
  mp_collider = std::make_unique<ColliderType>(std::forward<Args>(i_args)...);
  mp_collider->SetTransformationSource(m_transformation);
}

template <class PhysicalProperty,
          class... Args,
          std::enable_if_t<std::is_base_of_v<PhysicalProperty2D, PhysicalProperty>, bool>>
void Object2D::InitPhysicalProperty(Args&&... i_args)
{
  mp_physical_properties = std::make_unique<PhysicalProperty>(std::forward<Args>(i_args)...);
  mp_physical_properties->SetTransformationSource(m_transformation);
}
