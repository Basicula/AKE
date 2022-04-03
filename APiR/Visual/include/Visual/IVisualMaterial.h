#pragma once
#include "Visual/Color.h"

// Material that will be using in rendering workflow
// Tells how we see the object in the world
// For example color, diffuse, reflected, refracted etc
class IVisualMaterial
  {
  public:
    virtual ~IVisualMaterial() = default;

    virtual Color GetPrimitiveColor() const = 0;

    virtual Color CalculateColor(
      const Vector3d& i_normal,
      const Vector3d& i_view_direction,
      const Vector3d& i_light_direction) const = 0;

    // number in interval [0,1]
    // where 0 means no reflection
    // and 1 for full reflection (impossible in real world)
    virtual bool IsReflectable() const = 0;
    virtual double ReflectionInfluence() const = 0;
    virtual Vector3d ReflectedDirection(
      const Vector3d& i_normal_at_point, 
      const Vector3d& i_view_direction) const = 0;

    virtual bool IsRefractable() const = 0;
    virtual double RefractionInfluence() const = 0;
    virtual Vector3d RefractedDirection() const = 0;
  };