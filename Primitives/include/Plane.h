#pragma once

#include <Vector.h>
#include <IObject.h>

class Plane : public IObject
{
public:
  Plane() = delete;
  Plane(const Vector3d& i_first, const Vector3d& i_second, const Vector3d& i_third);
  Plane(const Vector3d& i_point, const Vector3d& i_normal);

  bool GetNormalInPoint(Vector3d& o_normal,const Vector3d& i_point) const override;
  Vector3d GetNormal() const;

  int WhereIsPoint(const Vector3d& i_point) const;

  double* GetCoefs() const;

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
