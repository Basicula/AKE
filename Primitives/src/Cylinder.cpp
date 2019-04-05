#include <Cylinder.h>
#include <SolveEquations.h>

Cylinder::Cylinder(const Vector3d& i_center, double i_radius, double i_height, const ColorMaterial& i_material)
  : IObject(i_material, ObjectType::CYLINDER)
  , m_center(i_center)
  , m_radius(i_radius)
  , m_height(i_height)
  , m_is_finite(i_height > 0)
  {}

bool Cylinder::GetNormalInPoint(Vector3d& o_normal, const Vector3d& i_point) const
  {
  Vector3d point = i_point - m_center;
  if (abs(point[0] * point[0] + point[1] * point[1] - m_radius * m_radius) < 1e-10)
    o_normal = Vector3d(point[0] / m_radius, point[1] / m_radius, 0);
  else
    o_normal = Vector3d(0, 0, point[2] > 0 ? 1 : -1);
  return true;
  }

bool Cylinder::IntersectWithRay(Vector3d& o_intersecion, double &o_distance, const Ray& i_ray) const
  {
  Vector3d direction = i_ray.GetDirection();
  Vector3d origin = i_ray.GetStart() - m_center;
  const double zmin = -m_height / 2;
  const double zmax = m_height / 2;
  double quadratic_coefs[3] = {
    origin[0] * origin[0] + origin[1] * origin[1] - m_radius * m_radius,
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
    if (roots[0] < Epsilon3D)
      return false;
    Vector3d intersection = origin + direction * roots[0];
    if (intersection[2] >= zmax || intersection[2] <= zmin)
      return false;
    o_distance = roots[0];
    o_intersecion = intersection + m_center;
    return true;
    }
  else
    {
    if (roots[0] < Epsilon3D && roots[1] < Epsilon3D)
      return false;
    if (roots[0] > roots[1])
      std::swap(roots[0], roots[1]);
    Vector3d first_intersection = origin + direction * roots[0];
    Vector3d second_intersection = origin + direction * roots[1];
    if (!m_is_finite || first_intersection[2] <= zmax && first_intersection[2] >= zmin && roots[0] > Epsilon3D)
      {
      o_distance = roots[0];
      o_intersecion = first_intersection + m_center;
      return true;
      }
    if ((first_intersection[2] > zmax && second_intersection[2] <= zmax) ||
      (first_intersection[2] < zmin && second_intersection[2] >= zmin))
      {
      double dz = first_intersection[2] > zmax ? first_intersection[2] - zmax : zmin - first_intersection[2];
      double dist = first_intersection.Distance(second_intersection);
      double to_intersection = dz * dist / abs(first_intersection[2] - second_intersection[2]);
      if (roots[0] + to_intersection < Epsilon3D)
        return false;
      o_distance = roots[0] + to_intersection;
      o_intersecion = i_ray.GetStart() + direction * o_distance;
      return true;
      }
    }
  return false;
  }