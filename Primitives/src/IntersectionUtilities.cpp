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
  if (distance_to_intersection <= 0)
    return false;
  o_intersection = ray_start + i_ray.GetDirection()*distance_to_intersection;
  return true;
  }

bool IntersectRayWithPlane(Vector3d & o_intersection, const Ray & i_ray, const Plane & i_plane)
  {
  if (i_plane.WhereIsPoint(i_ray.GetStart()) * i_ray.GetDirection().Dot(i_plane.GetNormal()) >= 0)
    return false;
  double distance_to_intersection = abs((i_plane.GetNormal().Dot(i_ray.GetStart()) + i_plane.GetD()) / i_plane.GetNormal().Dot(i_ray.GetDirection()));
  if(distance_to_intersection < 1e-16)
    return false;
  o_intersection = i_ray.GetStart() + i_ray.GetDirection()*distance_to_intersection;
  return true;
  }

bool IntersectRayWithWave(Vector3d& o_intersection, const Ray& i_ray)
  {
  // z + a = sin(x^2 + y^2)
  // z + a = sin(sqrt(x^2+y^2))
  const double PI = 3.14159265359;
  double tr = 1000, tl = 0;
  const double a = -210;
  const double eps = 1e-6;
  Vector3d far_point = i_ray.GetStart() + i_ray.GetDirection() * tr;
  double eqr = far_point[2] + a - sin(far_point[0] * far_point[0] + far_point[1] * far_point[1]);
  double eql = i_ray.GetStart()[2] + a - sin(i_ray.GetStart()[0] * i_ray.GetStart()[0] + i_ray.GetStart()[1] * i_ray.GetStart()[1]);
  if (eqr*eql > 0)
    return false;
  while (true)
    {
    double tm = (tr + tl) / 2;
    Vector3d probably_intersection = i_ray.GetStart() + i_ray.GetDirection() * tm;
    double x = probably_intersection[0] - PI * int(probably_intersection[0] / PI);
    double y = probably_intersection[1] - PI * int(probably_intersection[1] / PI);
    //double sqrx = x * x;
    double sqrx = probably_intersection[0] * probably_intersection[0];
    //double sqry = y * y;
    double sqry = probably_intersection[1] * probably_intersection[1];
    double eq = probably_intersection[2] + a - sin(sqrt(sqrx + sqry));
    if (abs(eq) <= eps)
      {
      o_intersection = probably_intersection;
      return true;
      }
    else if (eq > 0)
      tr = tm;
    else
      tl = tm;
    }
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