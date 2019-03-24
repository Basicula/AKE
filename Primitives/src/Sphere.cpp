#include <Sphere.h>

Sphere::Sphere(const ColorMaterial& i_material)
  : IObject(i_material, ObjectType::SPHERE)
  , m_center(Vector3d(0, 0, 0))
  , m_radius(1)
  {}

Sphere::Sphere(double i_radius, const ColorMaterial& i_material)
  : IObject(i_material, ObjectType::SPHERE)
  , m_center(Vector3d(0, 0, 0))
  , m_radius(i_radius)
  {}

Sphere::Sphere(const Vector3d& i_center, double i_radius, const ColorMaterial& i_material)
  : IObject(i_material, ObjectType::SPHERE)
  , m_center(i_center)
  , m_radius(i_radius)
  {}

bool Sphere::GetNormalInPoint(Vector3d& o_normal, const Vector3d& i_point) const
  {
  o_normal = (i_point - m_center).Normalized();
  return true;
  }