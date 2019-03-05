#pragma once

#include <Vector.h>

class Sphere
{
public:
  Sphere();
  Sphere(double i_radius);
  Sphere(const Vector3d& i_center, double i_radius);

  inline Vector3d GetCenter() const { return m_center; };
  inline double GetRadius() const { return m_radius; };
private:
  Vector3d m_center;
  double m_radius;
};