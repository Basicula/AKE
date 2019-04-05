#include "Plane.h"

Plane::Plane(const Vector3d& i_first, const Vector3d& i_second, const Vector3d& i_third, const ColorMaterial& i_material)
  : IObject(i_material, ObjectType::PLANE)
  , m_point(i_first)
  , m_normal((i_second - i_first).CrossProduct(i_third - i_first))
  {
  _FillParams();
  }

Plane::Plane(const Vector3d& i_point, const Vector3d& i_normal, const ColorMaterial& i_material)
  : IObject(i_material, ObjectType::PLANE)
  , m_point(i_point)
  , m_normal(i_normal)
  {
  _FillParams();
  }

void Plane::_FillParams()
  {
  m_a = m_normal[0];
  m_b = m_normal[1];
  m_c = m_normal[2];
  m_d = -m_normal.Dot(m_point);
  }

Vector3d Plane::GetNormal() const
  {
  return m_normal;
  }

bool Plane::GetNormalInPoint(Vector3d& o_normal, const Vector3d& i_point) const
  {
  o_normal = m_normal;
  return true;
  }

double Plane::GetValueFromEquation(const Vector3d& i_point) const
  {
  return i_point.Dot(m_normal) + m_d;
  }


bool Plane::IntersectWithRay(Vector3d & o_intersection, double &o_distance, const Ray & i_ray) const
  {
  const double value_from_equation = GetValueFromEquation(i_ray.GetStart());
  if (value_from_equation * i_ray.GetDirection().Dot(m_normal) >= 0)
    return false;
  double distance_to_intersection = abs(value_from_equation / m_normal.Dot(i_ray.GetDirection()));
  if (distance_to_intersection <= Epsilon3D)
    return false;
  o_distance = distance_to_intersection;
  o_intersection = i_ray.GetStart() + i_ray.GetDirection()*distance_to_intersection;
  return true;
  }
