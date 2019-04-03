#include <math.h>

#include <Torus.h>

Torus::Torus(const Vector3d& i_center, double i_major_radius, double i_minor_radius, const ColorMaterial& i_material)
  : IObject(i_material, ObjectType::TORUS)
  , m_center(i_center)
  , m_major_radius(i_major_radius)
  , m_minor_radius(i_minor_radius)
  {}

bool Torus::GetNormalInPoint(Vector3d& o_normal, const Vector3d& i_point) const
  {
  Vector3d point = i_point - m_center;
  double sqrt_sum_sqr = sqrt(point[0] * point[0] + point[1] * point[1]);
  double dx = 2 * point[0] * (1 - m_major_radius / sqrt_sum_sqr);
  double dy = 2 * point[1] * (1 - m_major_radius / sqrt_sum_sqr);
  double dz = 2 * point[2];
  o_normal = Vector3d(dx, dy, dz).Normalized();
  return true;
  }