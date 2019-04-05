#include <Cylinder.h>
#include<iostream>
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