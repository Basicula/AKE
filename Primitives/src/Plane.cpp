#include "Plane.h"

Plane::Plane(const Vector3d& i_first, const Vector3d& i_second, const Vector3d& i_third)
  : m_point(i_first)
  , m_normal((i_second - i_first).CrossProduct(i_third - i_first))
{
  _FillParams();
}

Plane::Plane(const Vector3d& i_point, const Vector3d& i_normal)
  : m_point(i_point)
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

inline Vector3d Plane::GetNormal() const
{
  return m_normal;
}
