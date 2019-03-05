#pragma once

#include <Vector.h>

class Plane
{
public:
  Plane() = delete;
  Plane(const Vector3d& i_first, const Vector3d& i_second, const Vector3d& i_third);
  Plane(const Vector3d& i_point, const Vector3d& i_normal);

  inline Vector3d GetNormal() const;

private:
  void _FillParams();

private:
  Vector3d m_point;
  Vector3d m_normal;

  double m_a;
  double m_b;
  double m_c;
  double m_d;

};
