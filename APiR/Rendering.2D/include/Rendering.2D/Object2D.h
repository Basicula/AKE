#pragma once
#include "Geometry.2D/Shape.h"
#include "Geometry/Transformation.h"
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

  [[nodiscard]] const Transformation2D& GetTransformation() const;
  [[nodiscard]] Transformation2D& GetTransformation();

private:
  std::unique_ptr<Shape> mp_shape;
  std::unique_ptr<Drawer> mp_drawer;
  Transformation2D m_transformation;
};

#include "impl/Object2DImpl.h"