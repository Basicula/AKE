#pragma once
#include "Geometry.2D/Shape2D.h"
#include "Geometry/Transformation.h"
#include "Physics.2D/GJKConvex2D.h"
#include "Rendering.2D/SimpleDrawer.h"

#include <memory>

class Object2D
{
public:
  template <class ShapeType, class... Args>
  void InitShape(Args&&... i_args);
  [[nodiscard]] const Shape2D& GetShape() const;

  template <class DrawerType, class... Args>
  void InitDrawer(Args&&... i_args);
  [[nodiscard]] const SimpleDrawer& GetDrawer() const;
  [[nodiscard]] SimpleDrawer& GetDrawer();

  template <class ColliderType, class... Args>
  void InitCollider(Args&&... i_args);
  [[nodiscard]] const GJKConvex2D* GetCollider() const;

  [[nodiscard]] const Transformation2D& GetTransformation() const;
  [[nodiscard]] Transformation2D& GetTransformation();

private:
  std::unique_ptr<Shape2D> mp_shape;
  std::unique_ptr<SimpleDrawer> mp_drawer;
  std::unique_ptr<GJKConvex2D> mp_collider;
  Transformation2D m_transformation;
};

#include "impl/Object2DImpl.h"