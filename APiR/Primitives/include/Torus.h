#pragma once

#include <Vector.h>
#include <ISurface.h>

//based on torus equation
/*
z^2 = a^2 - (b - sqrt(x^2 + y^2))^2
i.e.
z^2 = m_minor_radius - (m_major_radius - sqrt(x^2+y^2))^2
*/
class Torus : public ISurface
  {
  public:
    Torus() = delete;
    Torus(
      const Vector3d& i_center, 
      double i_major_radius, 
      double i_minor_radius);

    Vector3d GetCenter() const;
    double GetMinor() const;
    double GetMajor() const;

    virtual std::string Serialize() const override;
  protected:
    virtual BoundingBox _GetBoundingBox() const override;
    virtual Vector3d _NormalAtPoint(const Vector3d& i_point) const override;
    virtual bool _IntersectWithRay(
      IntersectionRecord& io_intersection, 
      const Ray& i_ray) const override;

  private:
    void _CalculateBoundingBox();

  private:
    Vector3d m_center;
    double m_major_radius;
    double m_minor_radius;
    BoundingBox m_bounding_box;
  };

inline Vector3d Torus::GetCenter() const 
  { 
  return m_center; 
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
