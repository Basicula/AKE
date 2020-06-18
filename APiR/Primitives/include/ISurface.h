#pragma once
#include <string>
#include <memory>

#include <BoundingBox.h>
#include <Intersection.h>
#include <Ray.h>

class ISurface
  {
  public:
    ISurface() = default;
    virtual ~ISurface() = default;

    BoundingBox GetBoundingBox() const;
    bool IntersectWithRay(
      IntersectionRecord& o_intersection,
      const Ray& i_ray) const;
    Vector3d NormalAtPoint(const Vector3d& i_point) const;

    virtual std::string Serialize() const = 0;
  protected:
    virtual BoundingBox _GetBoundingBox() const = 0;
    virtual bool _IntersectWithRay(
      IntersectionRecord& o_intersection,
      const Ray& i_ray) const = 0;
    virtual Vector3d _NormalAtPoint(const Vector3d& i_point) const = 0;
  };

inline BoundingBox ISurface::GetBoundingBox() const
  {
  return _GetBoundingBox();
  }

inline bool ISurface::IntersectWithRay(
  IntersectionRecord& o_intersection,
  const Ray& i_ray) const
  {
  return _IntersectWithRay(o_intersection, i_ray);
  }

inline Vector3d ISurface::NormalAtPoint(const Vector3d& i_point) const
  {
  return _NormalAtPoint(i_point);
  }

using ISurfaceSPtr = std::shared_ptr<ISurface>;
using ISurfaceUPtr = std::unique_ptr<ISurface>;