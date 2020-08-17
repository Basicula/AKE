#pragma once
#include <Common/IObject.h>
#include <Geometry/Intersection.h>
#include <Geometry/Ray.h>

class IRenderable : public IObject
  {
  public:
    virtual ~IRenderable() = default;

    virtual bool IntersectWithRay(
      IntersectionRecord& o_intersection,
      const Ray& i_ray) const = 0;

    virtual BoundingBox GetBoundingBox() const = 0;

    virtual void Update() = 0;
  };

using IRenderableSPtr = std::shared_ptr<IRenderable>;
using IRenderableUPtr = std::unique_ptr<IRenderable>;
