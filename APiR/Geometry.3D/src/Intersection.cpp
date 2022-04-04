#include "Geometry.3D/Intersection.h"

#include "Common/Constants.h"
#include "Geometry/BoundingBox.h"
#include "Geometry.3D/Ray.h"

namespace {
  void RayBoxIntersection(bool& o_is_intersected,
                          double& o_near,
                          double& o_far,
                          const Ray& i_ray,
                          const BoundingBox& i_box)
  {
    const auto& min_corner = i_box.GetMin();
    const auto& max_corner = i_box.GetMax();
    const auto& origin = i_ray.GetOrigin();
    const auto& ray_direction = i_ray.GetDirection();

    o_near = -MAX_DOUBLE;
    o_far = MAX_DOUBLE;
    for (auto i = 0; i < 3; ++i) {
      const double inv_dir = 1.0 / ray_direction[i];
      const double t1 = (min_corner[i] - origin[i]) * inv_dir;
      const double t2 = (max_corner[i] - origin[i]) * inv_dir;
      const bool cond = (t1 < t2);
      const double ttmin = cond ? t1 : t2;
      const double ttmax = cond ? t2 : t1;

      if (ttmin > o_near)
        o_near = ttmin;
      if (ttmax < o_far)
        o_far = ttmax;

      if (o_near > o_far) {
        o_is_intersected = false;
        return;
      }
    }
    o_is_intersected = true;
  }
}

RayBoxIntersectionRecord::RayBoxIntersectionRecord()
  : m_intersected(false)
  , m_tmax(-MAX_DOUBLE)
  , m_tmin(MAX_DOUBLE)
{}

void RayBoxIntersectionRecord::Reset()
{
  m_intersected = false;
  m_tmax = -MAX_DOUBLE;
  m_tmin = MAX_DOUBLE;
}

void RayBoxIntersection(const Ray& i_ray, const BoundingBox& i_box, RayBoxIntersectionRecord& o_intersection)
{
  RayBoxIntersection(o_intersection.m_intersected, o_intersection.m_tmin, o_intersection.m_tmax, i_ray, i_box);
}

bool RayIntersectBox(const Ray& i_ray, const BoundingBox& i_box)
{
  bool is_intersected;
  double temp_near, temp_far;
  RayBoxIntersection(is_intersected, temp_near, temp_far, i_ray, i_box);
  return is_intersected;
}