#pragma once
#include <Visual/Color.h>
#include <Visual/ILight.h>

#include <string>
#include <memory>

class IMaterial
  {
  public:
    virtual std::string Serialize() const = 0;
    virtual Color GetPrimitiveColor() const = 0;

    virtual Color GetLightInfluence(
      const Vector3d& i_point,
      const Vector3d& i_normal,
      const Vector3d& i_view_direction,
      std::shared_ptr<ILight> i_light) const = 0;

    // number in interval [0,1]
    // where 0 means no reflection
    // and 1 for full reflection (impossible in real world)
    virtual double ReflectionInfluence() const = 0;
    virtual bool IsReflectable() const = 0;
    virtual Vector3d ReflectedDirection(
      const Vector3d& i_normal_at_point, 
      const Vector3d& i_view_direction) const = 0;

    virtual double RefractionInfluence() const = 0;
    virtual bool IsRefractable() const = 0;
    virtual Vector3d RefractedDirection() const = 0;

    virtual ~IMaterial() = default;
  };

using IMaterialSPtr = std::shared_ptr<IMaterial>;
using IMaterialUPtr = std::unique_ptr<IMaterial>;