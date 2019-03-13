#include "Plane.h"

Plane::Plane(const Vector3d& i_first, const Vector3d& i_second, const Vector3d& i_third)
  : IObject(ObjectType::PLANE)
  , m_point(i_first)
  , m_normal((i_second - i_first).CrossProduct(i_third - i_first))
  {
  _FillParams();
  }

Plane::Plane(const Vector3d& i_point, const Vector3d& i_normal)
  : IObject(ObjectType::PLANE)
  , m_point(i_point)
  , m_normal(i_normal)
  {
  _FillParams();
  }

void Plane::_FillParams()
  {
  double* coords = m_normal.GetArray();
  m_a = coords[0];
  m_b = coords[1];
  m_c = coords[2];
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

double* Plane::GetCoefs() const
  {
  return new double[4]{ m_a, m_b, m_c, m_d };
  }