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

int Plane::WhereIsPoint(const Vector3d& i_point) const
  {
  double eq_res = i_point.Dot(m_normal) + m_d;
  return eq_res > 0 ? 1 : eq_res < 0 ? -1 : 0;
  }