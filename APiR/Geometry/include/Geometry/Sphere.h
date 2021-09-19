#pragma once
#include <Geometry/ISurface.h>

class Sphere : public ISurface
  {
  public:
    Sphere(
      const Vector3d& i_center, 
      double i_radius);

    virtual std::string Serialize() const override;

    Vector3d GetCenter() const;
    void SetCenter(const Vector3d& i_center);

    double GetRadius() const;
    void SetRadius(double i_radius);

  protected:
    virtual void _CalculateBoundingBox() override;
    virtual bool _IntersectWithRay(
      double& o_intersection_dist,
      const Ray& i_ray,
      const double i_far) const override;
    virtual Vector3d _NormalAtLocalPoint(const Vector3d& i_point) const override;

  private:
    double m_radius;
  };

inline Vector3d Sphere::GetCenter() const 
  { 
  return GetTranslation(); 
  };

inline void Sphere::SetCenter(const Vector3d& i_center)
  {
  SetTranslation(i_center);
  };

inline double Sphere::GetRadius() const 
  { 
  return m_radius; 
  };

inline void Sphere::SetRadius(double i_radius)
  {
  m_radius = i_radius;
  };

inline std::string Sphere::Serialize() const
  {
  std::string res = "{ \"Sphere\" : { ";
  res += "\"Center\" : " + GetCenter().Serialize() + ", ";
  res += "\"Radius\" : " + std::to_string(m_radius);
  res += "} }";
  return res;
  }

inline Vector3d Sphere::_NormalAtLocalPoint(const Vector3d& i_local_point) const
  {
  return i_local_point.Normalized();
  }