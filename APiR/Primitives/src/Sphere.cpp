#include <Sphere.h>
#include <DefinesAndConstants.h>
#include <SolveEquations.h>

Sphere::Sphere(const Vector3d& i_center, double i_radius)
  : m_center(i_center)
  , m_radius(i_radius)
  {}

bool Sphere::_IntersectWithRay(
  IntersectionRecord& o_intersection,
  const Ray& i_ray) const
  {
  const auto& ray_origin = i_ray.GetOrigin();
  const auto& ray_direction = i_ray.GetDirection();
  
  const double distance_to_sphere = (m_center - ray_origin).Dot(ray_direction);
  
  if (distance_to_sphere - m_radius > o_intersection.m_distance)
    return false;
  
  const auto& point_near_sphere = ray_origin + ray_direction * distance_to_sphere;
  
  const double sqr_radius = m_radius * m_radius;
  const double center_to_point = m_center.SquareDistance(point_near_sphere);
  if (center_to_point > sqr_radius)
    return false;
  
  const double distance = distance_to_sphere - sqrt(sqr_radius - center_to_point);
  if (distance <= 0.0)
    return false;
  
  if (o_intersection.m_distance < distance)
    return false;
  
  o_intersection.m_distance = distance;
  o_intersection.m_intersection = ray_origin + ray_direction * distance;
  o_intersection.m_normal = NormalAtPoint(o_intersection.m_intersection);
  return true;
  }