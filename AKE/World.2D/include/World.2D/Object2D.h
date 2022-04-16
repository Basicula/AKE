#pragma once
#include "Geometry.2D/Shape2D.h"
#include "Geometry/Transformation.h"
#include "Physics.2D/GJKConvex2D.h"
#include "Rendering.2D/SimpleDrawer.h"

#include <memory>

class Object2D
{
public:
  template <class ShapeType, class... Args, std::enable_if_t<std::is_base_of_v<Shape2D, ShapeType>, bool> = true>
  void InitShape(Args&&... i_args);
  [[nodiscard]] const Shape2D& GetShape() const;

  template <class DrawerType, class... Args, std::enable_if_t<std::is_base_of_v<SimpleDrawer, DrawerType>, bool> = true>
  void InitDrawer(Args&&... i_args);
  [[nodiscard]] const SimpleDrawer& GetDrawer() const;
  [[nodiscard]] SimpleDrawer& GetDrawer();

  template <class ColliderType,
            class... Args,
            std::enable_if_t<std::is_base_of_v<GJKConvex2D, ColliderType>, bool> = true>
  void InitCollider(Args&&... i_args);
  [[nodiscard]] const GJKConvex2D* GetCollider() const;

  template <class PhysicalProperty,
            class... Args,
            std::enable_if_t<std::is_base_of_v<PhysicalProperty2D, PhysicalProperty>, bool> = true>
  void InitPhysicalProperty(Args&&... i_args);
  [[nodiscard]] const PhysicalProperty2D* GetPhysicalProperty() const;
  [[nodiscard]] PhysicalProperty2D* GetPhysicalProperty();

  [[nodiscard]] const Transformation2D& GetTransformation() const;
  [[nodiscard]] Transformation2D& GetTransformation();

private:
  std::unique_ptr<Shape2D> mp_shape;
  std::unique_ptr<SimpleDrawer> mp_drawer;
  std::unique_ptr<GJKConvex2D> mp_collider;
  std::unique_ptr<PhysicalProperty2D> mp_physical_properties;
  Transformation2D m_transformation;
};

#include "impl/Object2DImpl.h"