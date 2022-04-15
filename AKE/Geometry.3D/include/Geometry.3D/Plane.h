#pragma once
#include "Geometry.3D/ISurface.h"
#include "Math/Vector.h"

class Plane : public ISurface
{
public:
  Plane() = delete;
  Plane(const Vector3d& i_first, const Vector3d& i_second, const Vector3d& i_third);
  Plane(const Vector3d& i_point, const Vector3d& i_normal);

  Vector3d GetNormal() const;

  virtual std::string Serialize() const override;

protected:
  virtual void _CalculateBoundingBox() override;
  virtual bool _IntersectWithRay(double& io_nearest_intersection_dist,
                                 const Ray& i_ray,
                                 const double i_far) const override;
  virtual Vector3d _NormalAtLocalPoint(const Vector3d& i_point) const override;

private:
  void _FillParams();

private:
  Vector3d m_normal;

  double m_a;
  double m_b;
  double m_c;

  double m_size_limit;
};

inline Vector3d Plane::_NormalAtLocalPoint(const Vector3d& /*i_point*/) const
{
  return m_normal;
}

inline Vector3d Plane::GetNormal() const
{
  return m_normal;
}

inline std::string Plane::Serialize() const
{
  return std::string();
}