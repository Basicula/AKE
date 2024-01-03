#pragma once
#include "Common/Constants.h"
#include "Geometry.3D/Ray.h"
#include "Geometry.3D/Transformable.h"
#include "Geometry/BoundingBox.h"

class ISurface : public Transformable
{
public:
  ISurface() = default;
  virtual ~ISurface() = default;

  BoundingBox3D GetBoundingBox() const;
  bool IntersectWithRay(double& o_distance, const Ray& i_ray, double i_far = Common::Constants::MAX_DOUBLE) const;
  Vector3d NormalAtPoint(const Vector3d& i_point) const;

  virtual std::string Serialize() const = 0;

protected:
  // used to store bbox once for further manipulation
  // and for changing by need
  virtual void _CalculateBoundingBox() = 0;
  virtual bool _IntersectWithRay(double& o_intersection_dist, const Ray& i_local_ray, double i_far) const = 0;
  virtual Vector3d _NormalAtLocalPoint(const Vector3d& i_local_point) const = 0;

  void _OnTransformationChange() override;

protected:
  BoundingBox3D m_bounding_box;
};