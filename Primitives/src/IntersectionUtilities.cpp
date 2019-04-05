#include "IntersectionUtilities.h"

#include <vector>

#include <SolveEquations.h>

const double EPS = 1e-10;

bool IntersectRayWithSphere(Vector3d& o_intersection, double &o_distance, const Ray& i_ray, const Sphere& i_sphere)
  {
  Vector3d sphere_center = i_sphere.GetCenter();
  Vector3d ray_start = i_ray.GetStart();
  Vector3d start_to_center = sphere_center - ray_start;

  double to_sphere = start_to_center.Dot(i_ray.GetDirection());
  Vector3d point_near_sphere = ray_start + i_ray.GetDirection()*to_sphere;

  const double sqr_radius = i_sphere.GetRadius() * i_sphere.GetRadius();
  double sphere_center_to_point = sphere_center.SquareDistance(point_near_sphere);
  if (sphere_center_to_point > sqr_radius)
    return false;

  double distance_to_intersection = to_sphere - sqrt(sqr_radius - sphere_center_to_point);
  if (distance_to_intersection <= EPS)
    return false;
  o_distance = distance_to_intersection;
  o_intersection = ray_start + i_ray.GetDirection()*distance_to_intersection;
  return true;
  }

bool IntersectRayWithPlane(Vector3d & o_intersection, double &o_distance, const Ray & i_ray, const Plane & i_plane)
  {
  if (i_plane.WhereIsPoint(i_ray.GetStart()) * i_ray.GetDirection().Dot(i_plane.GetNormal()) >= 0)
    return false;
  double distance_to_intersection = abs((i_plane.GetNormal().Dot(i_ray.GetStart()) + i_plane.GetD()) / i_plane.GetNormal().Dot(i_ray.GetDirection()));
  if (distance_to_intersection <= EPS)
    return false;
  o_distance = distance_to_intersection;
  o_intersection = i_ray.GetStart() + i_ray.GetDirection()*distance_to_intersection;
  return true;
  }

bool IntersectRayWithTorus(Vector3d & o_intersecion, double &o_distance, const Ray & i_ray, const Torus & i_torus)
  {
  /*
  Let's make intersection equation by putting ray point into torus equation
  Ray point : O + D*t = P
  Torus equation : (z^2 - minor^2 + major^2 + x^2 + y^2) = 4 * major^2 *(x^2 + y^2)
  in result we have equation c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0 = 0
  need to find t - distance to intersection
  */
  Vector3d start = i_ray.GetStart() - i_torus.GetCenter();
  double square_start = start.SquareLength();
  double start_dot_direction = i_ray.GetDirection().Dot(start);
  double square_minor = i_torus.GetMinor() * i_torus.GetMinor();
  double square_major = i_torus.GetMajor() * i_torus.GetMajor();
  double square_start_major_minor = square_start + square_major - square_minor;
  double c4 = 1;
  double c3 = 4 * start_dot_direction;
  double c2 = 4 * start_dot_direction * start_dot_direction
    + 2 * square_start_major_minor
    - 4 * square_major * (1 - i_ray.GetDirection()[2] * i_ray.GetDirection()[2]);
  double c1 = 4 * square_start_major_minor * start_dot_direction - 8 * square_major * (start_dot_direction - i_ray.GetDirection()[2] * start[2]);
  double c0 = square_start_major_minor * square_start_major_minor - 4 * square_major * (square_start - start[2] * start[2]);
  double roots[4] = { INFINITY,INFINITY ,INFINITY ,INFINITY };
  const double quartic_coefs[5] = { c0,c1,c2,c3,c4 };
  int roots_count = Equations::SolveQuartic(quartic_coefs, roots);
  if (roots_count == 0)
    return false;
  //find smallest positive if exist temp solution
  double distance = INFINITY;
  for (const auto& root : roots)
    if (root > EPS && root < distance)
      distance = root;
  if (distance == INFINITY)
    return false;
  o_distance = distance;
  o_intersecion = start + i_ray.GetDirection() * distance + i_torus.GetCenter();
  return true;
  }

bool IntersectRayWithCylinder(Vector3d& o_intersecion, double &o_distance, const Ray& i_ray, const Cylinder& i_torus)
  {
  Vector3d direction = i_ray.GetDirection();
  Vector3d origin = i_ray.GetStart() - i_torus.GetCenter();
  const double zmin = -i_torus.GetHeight() / 2;
  const double zmax = i_torus.GetHeight() / 2;
  double quadratic_coefs[3] = {
    origin[0] * origin[0] + origin[1] * origin[1] - i_torus.GetRadius() * i_torus.GetRadius(),
    2 * origin[0] * direction[0] + 2 * origin[1] * direction[1],
    direction[0] * direction[0] + direction[1] * direction[1] };
  double roots[2] = { INFINITY, INFINITY };
  int roots_count = Equations::SolveQuadratic(quadratic_coefs, roots);
  if (roots_count == 0)
    {
    return false;
    }
  if (roots_count == 1)
    {
    if (roots[0] < EPS)
      return false;
    Vector3d intersection = origin + direction * roots[0];
    if (intersection[2] >= zmax || intersection[2] <=zmin)
      return false;
    o_distance = roots[0];
    o_intersecion = intersection + i_torus.GetCenter();
    return true;
    }
  else
    {
    if (roots[0] < EPS && roots[1] < EPS)
      return false;
    if (roots[0] > roots[1])
      std::swap(roots[0], roots[1]);
    Vector3d first_intersection = origin + direction * roots[0];
    Vector3d second_intersection = origin + direction * roots[1];
    if (first_intersection[2] <= zmax && first_intersection[2] >= zmin && roots[0] > EPS)
      {
      o_distance = roots[0];
      o_intersecion = first_intersection + i_torus.GetCenter();
      return true;
      }
    if ((first_intersection[2] > zmax && second_intersection[2] <= zmax) ||
      (first_intersection[2] < zmin && second_intersection[2] >= zmin))
      {
      double dz = first_intersection[2] > zmax ? first_intersection[2] - zmax : zmin - first_intersection[2];
      double dist = first_intersection.Distance(second_intersection);
      double to_intersection = dz * dist / abs(first_intersection[2] - second_intersection[2]);
      if(roots[0] + to_intersection < EPS)
        return false;
      o_distance = roots[0] + to_intersection;
      o_intersecion = i_ray.GetStart() + direction * o_distance;
      return true;
      }
    }
  return false;
  }

bool IntersectRayWithWave(Vector3d& o_intersection, double &o_distance, const Ray& i_ray)
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

bool IntersectRayWithObject(Vector3d& o_intersection, double &o_distance, const Ray& i_ray, const IObject* ip_object)
  {
  switch (ip_object->GetType())
    {
    case IObject::ObjectType::SPHERE:
      return IntersectRayWithSphere(o_intersection, o_distance, i_ray, *static_cast<const Sphere*>(ip_object));
    case IObject::ObjectType::PLANE:
      return IntersectRayWithPlane(o_intersection, o_distance, i_ray, *static_cast<const Plane*>(ip_object));
    case IObject::ObjectType::TORUS:
      return IntersectRayWithTorus(o_intersection, o_distance, i_ray, *static_cast<const Torus*>(ip_object));
    case IObject::ObjectType::CYLINDER:
      return IntersectRayWithCylinder(o_intersection, o_distance, i_ray, *static_cast<const Cylinder*>(ip_object));
    default:
      break;
    }
  return true;
  }