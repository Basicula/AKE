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
  if (i_plane.WhereIsPoint(i_ray.GetStart()) * i_ray.GetDirection().Dot(i_plane.GetNormal()) >= 0)
    return false;
  double distance_to_intersection = abs((i_plane.GetNormal().Dot(i_ray.GetStart()) + i_plane.GetCoefs()[3]) / i_plane.GetNormal().Dot(i_ray.GetDirection()));
  o_intersection = i_ray.GetStart() + i_ray.GetDirection()*distance_to_intersection;
  return true;
  }

bool IntersectRayWithObject(Vector3d& o_intersection, const Ray& i_ray, const IObject* ip_object)
  {
  switch (ip_object->GetType())
    {
    case IObject::ObjectType::SPHERE:
      return IntersectRayWithSphere(o_intersection, i_ray, *dynamic_cast<const Sphere*>(ip_object));
      break;
    case IObject::ObjectType::PLANE:
      return IntersectRayWithPlane(o_intersection, i_ray, *dynamic_cast<const Plane*>(ip_object));
    default:
      break;
    }
  return true;
  }