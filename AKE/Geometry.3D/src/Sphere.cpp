#include "Geometry.3D/Sphere.h"

Sphere::Sphere(const Vector3d& i_center, double i_radius)
  : ISurface()
  , m_radius(i_radius)
{
  SetTranslation(i_center);
}

void Sphere::_CalculateBoundingBox()
{
  const Vector3d min_corner(-m_radius), max_corner(m_radius);
  m_bounding_box.AddPoint(min_corner);
  m_bounding_box.AddPoint(max_corner);
}

bool Sphere::_IntersectWithRay(double& o_intersection_dist, const Ray& i_local_ray, const double i_far) const
{
  // transform ray origin to sphere coordinate system
  // and solve sphere equation there
  const auto& ray_origin = i_local_ray.GetOrigin();
  const auto& ray_direction = i_local_ray.GetDirection();

  // dist to sphere = (center - ray origin).dot(ray direction)
  // after simplifications we have below value
  const double distance_to_sphere = -ray_origin.Dot(ray_direction);

  if (distance_to_sphere < 0.0 || distance_to_sphere - m_radius > i_far)
    return false;

  const auto& point_near_sphere = i_local_ray.GetPoint(distance_to_sphere);

  const double sqr_radius = m_radius * m_radius;
  // point near sphere now must be near coordinate origin
  // so we take square length as dist from sphere center to point
  const double center_to_point = point_near_sphere.SquareLength();
  if (center_to_point > sqr_radius)
    return false;

  const double half_horde = sqrt(sqr_radius - center_to_point);
  double distance = distance_to_sphere - half_horde;

  o_intersection_dist = distance;
  return true;
}