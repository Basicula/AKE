#include <Geometry/ISurface.h>

BoundingBox3D ISurface::GetBoundingBox() const {
  return m_bounding_box;
  }

bool ISurface::IntersectWithRay(
  double& o_distance,
  const Ray& i_ray,
  const double i_far) const {
  const auto& local_ray = RayToLocal(i_ray);
  return _IntersectWithRay(o_distance, local_ray, i_far);
  }

Vector3d ISurface::NormalAtPoint(const Vector3d& i_point) const {
  return _NormalAtLocalPoint(PointToLocal(i_point));
  }

void ISurface::_OnTransformationChange() {
  _CalculateBoundingBox();
  m_bounding_box = BBoxToWorld(m_bounding_box);
  }