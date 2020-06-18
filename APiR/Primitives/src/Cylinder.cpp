#include <Cylinder.h>
#include <SolveEquations.h>
#include <DefinesAndConstants.h>

Cylinder::Cylinder(
  const Vector3d& i_center,
  double i_radius,
  double i_height)
  : m_center(i_center)
  , m_radius(i_radius)
  , m_height(i_height)
  , m_is_finite(i_height > 0)
  {}

BoundingBox Cylinder::_GetBoundingBox() const
  {
  BoundingBox res;
  res.AddPoint(m_center + m_radius);
  res.AddPoint(m_center - m_radius);
  if (m_is_finite)
    {
    res.AddPoint(m_center + Vector3d(m_height / 2));
    res.AddPoint(m_center - Vector3d(m_height / 2));
    }
  else
    {
    res.AddPoint(m_center + Vector3d(0, 0, MAX_DOUBLE));
    res.AddPoint(m_center - Vector3d(0, 0, MAX_DOUBLE));
    }
  return res;
  }

Vector3d Cylinder::_NormalAtPoint(const Vector3d& i_point) const
  {
  Vector3d normal = i_point - m_center;
  if (m_is_finite && normal[2] == -m_height / 2)
    return Vector3d(0, 0, 1);

  if (m_is_finite && normal[2] == m_height / 2)
    return Vector3d(0, 0, -1);

  normal[2] = 0;
  return normal.Normalized();
  }

bool Cylinder::_IntersectWithRay(
  IntersectionRecord& io_intersection,
  const Ray& i_ray) const
  {
  const auto& ray_direction = i_ray.GetDirection();
  const auto& ray_origin = i_ray.GetOrigin();
  const auto to_cylinder_center = ray_origin - m_center;
  
  const double quadratic_coefs[3] =
    {
    to_cylinder_center[0] * to_cylinder_center[0] + to_cylinder_center[1] * to_cylinder_center[1] - m_radius * m_radius,
    2 * (to_cylinder_center[0] * ray_direction[0] + to_cylinder_center[1] * ray_direction[1]),
    ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1]
    };
  double roots[2] = { INFINITY, INFINITY };
  const int roots_count = Equations::SolveQuadratic(quadratic_coefs, roots);
  if (roots_count == 0)
    return false;
  
  const double zmin = -m_height / 2 + m_center[2];
  const double zmax = m_height / 2 + m_center[2];
  
  if (roots_count == 1)
    {
    Vector3d intersection = ray_origin + ray_direction * roots[0];
    if (intersection[2] >= zmax ||
        intersection[2] <= zmin ||
        roots[0] > io_intersection.m_distance ||
        roots[0] < 0.0)
      return false;
    io_intersection.m_distance = roots[0];
    io_intersection.m_intersection = intersection;
    io_intersection.m_normal = _NormalAtPoint(intersection);
    return true;
    }
  
  if (roots[0] < 0.0 && roots[1] < 0.0)
    return false;
  
  if (roots[0] > roots[1])
    std::swap(roots[0], roots[1]);
  
  Vector3d first_intersection = ray_origin + ray_direction * roots[0];
  if (!m_is_finite ||
      (first_intersection[2] <= zmax &&
       first_intersection[2] >= zmin))
    {
    if (roots[0] > io_intersection.m_distance)
      return false;
    io_intersection.m_distance = roots[0];
    io_intersection.m_intersection = first_intersection;
    io_intersection.m_normal = _NormalAtPoint(first_intersection);
    return true;
    }
  
  Vector3d second_intersection = ray_origin + ray_direction * roots[1];
  if ((first_intersection[2] > zmax && second_intersection[2] > zmax) ||
      (first_intersection[2] < zmin && second_intersection[2] < zmin))
    return false;
  
  const double dz = first_intersection[2] > zmax ? 
    first_intersection[2] - zmax : 
    zmin - first_intersection[2];
  const double dist = first_intersection.Distance(second_intersection);
  const double to_intersection = dz * dist / 
    (first_intersection[2] > second_intersection[2] ? 
     first_intersection[2] - second_intersection[2] : 
     second_intersection[2] - first_intersection[2]);
  const double distance = roots[0] + to_intersection;
  if (distance > io_intersection.m_distance)
    return false;
  io_intersection.m_distance = distance;
  io_intersection.m_intersection = ray_origin + ray_direction * distance;
  io_intersection.m_normal = first_intersection[2] > zmax ? Vector3d(0, 0, -1) : Vector3d(0, 0, 1);//_NormalAtPoint(io_intersection.m_intersection);
  return true;
  }