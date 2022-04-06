#pragma once
#include "Geometry.2D/Shape.h"
#include "Rendering.2D/Drawer.h"

#include <memory>

class Object2D
{
public:
  template <class ShapeType, class... Args>
  void InitShape(Args&&... i_args);
  [[nodiscard]] const Shape& GetShape() const;

  template <class DrawerType, class... Args>
  void InitDrawer(Args&&... i_args);
  [[nodiscard]] const Drawer& GetDrawer() const;

private:
  std::unique_ptr<Shape> mp_shape;
  std::unique_ptr<Drawer> mp_drawer;
};

#include "impl/Object2DImpl.h"