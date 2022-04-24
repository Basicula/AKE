#include "Utils2D/Utils2D.h"

#include "Common/Randomizer.h"
#include "Geometry.2D/Circle.h"
#include "Geometry.2D/Rectangle.h"
#include "Geometry.2D/Triangle2D.h"
#include "Math/Constants.h"
#include "Physics.2D/GJKCircleConvex2D.h"
#include "Physics.2D/GJKRectangleConvex2D.h"
#include "Physics.2D/GJKTriangleConvex2D.h"
#include "Rendering.2D/CircleDrawer.h"
#include "Rendering.2D/RectangleDrawer.h"
#include "Rendering.2D/Triangle2DDrawer.h"

namespace Utils2D {
  Vector2d RandomUnitVector(const double i_length)
  {
    const auto angle = Randomizer::Get(0.0, Math::Constants::TWOPI);
    const auto x = cos(angle);
    const auto y = sin(angle);
    return { i_length * x, i_length * y };
  }

  std::unique_ptr<Object2D> RandomObject()
  {
    enum class ObjectType : unsigned char
    {
      Circle = 0,
      Rectangle,
      Triangle,
      Undefined
    };
    const auto rand_object_type = static_cast<ObjectType>(
      Randomizer::Get(static_cast<unsigned char>(0), static_cast<unsigned char>(ObjectType::Undefined)));

    switch (rand_object_type) {
      case ObjectType::Circle:
        return RandomCircle(50, 100);
      case ObjectType::Rectangle:
        return RandomRectangle(50, 250);
      case ObjectType::Triangle:
        return RandomTriangle(100);
      case ObjectType::Undefined:
        return {};
    }
    return {};
  }

  std::unique_ptr<Object2D> RandomCircle(const double i_min_radius, const double i_max_radius)
  {
    auto p_object = std::make_unique<Object2D>();
    const auto radius = Randomizer::Get(i_min_radius, i_max_radius);
    p_object->InitShape<Circle>(radius);
    p_object->InitDrawer<CircleDrawer>(static_cast<const Circle&>(p_object->GetShape()), Color::RandomColor(), true);
    p_object->InitCollider<GJKCircleConvex2D>(radius);
    return p_object;
  }

  std::unique_ptr<Object2D> RandomRectangle(const double i_min_side_length, const double i_max_side_length)
  {
    auto p_object = std::make_unique<Object2D>();
    const auto width = Randomizer::Get(i_min_side_length, i_max_side_length);
    const auto height = Randomizer::Get(i_min_side_length, i_max_side_length);
    p_object->InitShape<Rectangle>(width, height);
    p_object->InitDrawer<RectangleDrawer>(
      static_cast<const Rectangle&>(p_object->GetShape()), Color::RandomColor(), true);
    p_object->InitCollider<GJKRectangleConvex2D>(width, height);
    return p_object;
  }

  std::unique_ptr<Object2D> RandomTriangle(const double i_max_side_length)
  {
    auto p_object = std::make_unique<Object2D>();
    const Vector2d vertices[] = { RandomUnitVector(i_max_side_length),
                                  RandomUnitVector(i_max_side_length),
                                  RandomUnitVector(i_max_side_length) };
    p_object->InitShape<Triangle2D>(vertices[0], vertices[1], vertices[2]);
    p_object->InitDrawer<Triangle2DDrawer>(
      static_cast<const Triangle2D&>(p_object->GetShape()), Color::RandomColor(), true);
    p_object->InitCollider<GJKTriangleConvex2D>(vertices[0], vertices[1], vertices[2]);
    return p_object;
  }
}
