#pragma once
#include <string>
#include <memory>

#include <BoundingBox.h>
#include <Intersection.h>
#include <Ray.h>
#include <Transformable.h>

class ISurface : public Transformable
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
    // used to store bbox once for further manipulation
    // and for changing by need
    virtual void _CalculateBoundingBox() = 0;
    virtual bool _IntersectWithRay(
      IntersectionRecord& o_intersection,
      const Ray& i_local_ray) const = 0;
    virtual Vector3d _NormalAtLocalPoint(const Vector3d& i_local_point) const = 0;

    virtual void _OnTransformationChange() override;

  protected:
    BoundingBox m_bounding_box;
  };

inline BoundingBox ISurface::GetBoundingBox() const
  {
  return m_bounding_box;
  }

inline bool ISurface::IntersectWithRay(
  IntersectionRecord& o_intersection,
  const Ray& i_ray) const
  {
  const auto local_ray = RayToLocal(i_ray);
  const bool is_intersected = _IntersectWithRay(o_intersection, local_ray);
  if (is_intersected)
    _LocalIntersectionToWorld(o_intersection);
  return is_intersected;
  }

inline Vector3d ISurface::NormalAtPoint(const Vector3d& i_local_point) const
  {
  return _NormalAtLocalPoint(i_local_point);
  }

inline void ISurface::_OnTransformationChange()
  {
  _CalculateBoundingBox();
  m_bounding_box = BBoxToWorld(m_bounding_box);
  }

using ISurfaceSPtr = std::shared_ptr<ISurface>;
using ISurfaceUPtr = std::unique_ptr<ISurface>;