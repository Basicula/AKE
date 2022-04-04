#pragma once
#include "Geometry/ISurface.h"
#include "Math/Vector.h"

// based on torus equation
/*
z^2 = a^2 - (b - sqrt(x^2 + y^2))^2
i.e.
z^2 = m_minor_radius - (m_major_radius - sqrt(x^2+y^2))^2
*/
class Torus : public ISurface
{
public:
  Torus() = delete;
  Torus(const Vector3d& i_center, double i_major_radius, double i_minor_radius);

  Vector3d GetCenter() const;
  double GetMinor() const;
  double GetMajor() const;

  virtual std::string Serialize() const override;

protected:
  virtual void _CalculateBoundingBox() override;
  virtual Vector3d _NormalAtLocalPoint(const Vector3d& i_point) const override;
  virtual bool _IntersectWithRay(double& o_intersection_dist, const Ray& i_ray, const double i_far) const override;

private:
  double m_major_radius;
  double m_minor_radius;
};

inline Vector3d Torus::GetCenter() const
{
  return GetTranslation();
}

inline double Torus::GetMinor() const
{
  return m_minor_radius;
}

inline double Torus::GetMajor() const
{
  return m_major_radius;
}

inline std::string Torus::Serialize() const
{
  return std::string();
}
