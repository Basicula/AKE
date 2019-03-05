#include "IntersectionUtilities.h"

bool IntersectRayWithSphere(Vector3d& o_intersection, const Ray& i_ray, const Sphere& i_sphere)
{
  Vector3d sphere_center = i_sphere.GetCenter();
  Vector3d ray_start = i_ray.GetStart();
  Vector3d start_to_center = sphere_center - ray_start;

  double to_sphere = start_to_center.Dot(i_ray.GetDirection());
  Vector3d point_near_sphere = ray_start + i_ray.GetDirection()*to_sphere;

  const double radius = i_sphere.GetRadius();
  double sphere_center_to_point = sphere_center.Distance(point_near_sphere);
  if (sphere_center_to_point > radius)
    return false;

  double distance_to_intersection = to_sphere - sqrt(radius*radius - sphere_center_to_point * sphere_center_to_point);
  o_intersection = ray_start + i_ray.GetDirection()*distance_to_intersection;
  return true;
}

bool IntersectRayWithPlane(Vector3d & o_intersection, const Ray & i_ray, const Plane & i_plane)
{
  return false;
  return true;
}
