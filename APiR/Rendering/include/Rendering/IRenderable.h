#pragma once
#include <Common/IObject.h>

#include <Geometry/BoundingBox.h>
#include <Geometry/Ray.h>

#include <Visual/IMaterial.h>

#include <memory>

class IRenderable : public IObject
  {
  public:
    virtual ~IRenderable() = default;

    virtual bool IntersectWithRay(
      double& o_distance,
      const Ray& i_ray,
      const double i_far) const = 0;

    virtual BoundingBox GetBoundingBox() const = 0;

    // for rendering often also need normal at intersection point
    virtual Vector3d GetNormalAtPoint(const Vector3d& i_point) const = 0;

    // renderable onject can't be rendered without material it's just have no sense
    virtual const IMaterial* GetMaterial() const = 0;
  };
