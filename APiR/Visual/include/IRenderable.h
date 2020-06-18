#pragma once

#include <IObject.h>
#include <Intersection.h>
#include <Ray.h>

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
