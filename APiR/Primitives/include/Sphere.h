#pragma once
#include <memory>

#include <ISurface.h>

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
    virtual BoundingBox _GetBoundingBox() const override;
    virtual bool _IntersectWithRay(
      IntersectionRecord& o_intersection,
      const Ray& i_ray) const override;
    virtual Vector3d _NormalAtPoint(const Vector3d& i_point) const override;

  private:
    Vector3d m_center;
    double m_radius;
  };

inline Vector3d Sphere::GetCenter() const 
  { 
  return m_center; 
  };

inline void Sphere::SetCenter(const Vector3d& i_center)
  {
  m_center = i_center;
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
  res += "\"Center\" : " + m_center.Serialize() + ", ";
  res += "\"Radius\" : " + std::to_string(m_radius);
  res += "} }";
  return res;
  }

inline BoundingBox Sphere::_GetBoundingBox() const
  {
  return BoundingBox(m_center - m_radius, m_center + m_radius);
  }

inline Vector3d Sphere::_NormalAtPoint(const Vector3d& i_point) const
  {
  return (i_point - m_center).Normalized();
  }