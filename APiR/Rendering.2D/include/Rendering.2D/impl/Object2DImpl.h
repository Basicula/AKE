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
}
