#pragma once
#include "Geometry/BoundingBox.h"
#include "Geometry/Ray.h"
#include "Geometry/Transformable.h"

#include "Common/Constants.h"

#include <memory>

class ISurface : public Transformable
  {
  public:
    ISurface() = default;
    virtual ~ISurface() = default;

    BoundingBox GetBoundingBox() const;
    bool IntersectWithRay(
      double& o_distance,
      const Ray& i_ray,
      const double i_far = MAX_DOUBLE) const;
    Vector3d NormalAtPoint(const Vector3d& i_point) const;

    virtual std::string Serialize() const = 0;
  protected:
    // used to store bbox once for further manipulation
    // and for changing by need
    virtual void _CalculateBoundingBox() = 0;
    virtual bool _IntersectWithRay(
      double& o_intersection_dist,
      const Ray& i_local_ray,
      const double i_far) const = 0;
    virtual Vector3d _NormalAtLocalPoint(const Vector3d& i_local_point) const = 0;

    virtual void _OnTransformationChange() override;

  protected:
    BoundingBox m_bounding_box;
  };