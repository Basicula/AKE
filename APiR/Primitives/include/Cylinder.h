#pragma once

#include <ISurface.h>
#include <Vector.h>

class Cylinder : public ISurface
  {
  public:
    Cylinder() = delete;
    //if i_height < 0 let cylinder be infinite
    //-1 default value for infiniteness
    Cylinder(
      const Vector3d& i_center, 
      double i_radius, 
      double i_height = -1);

    double GetRadius() const;
    Vector3d GetCenter() const;
    double GetHeight() const;

    bool IsFinite() const;
    void SetFiniteness(bool i_is_finite);

    virtual std::string Serialize() const override;
  protected:
    virtual BoundingBox _GetBoundingBox() const override;
    virtual Vector3d _NormalAtPoint(const Vector3d& i_point) const override;
    virtual bool _IntersectWithRay(
      IntersectionRecord& io_intersection, 
      const Ray& i_ray) const override;

  private:
    Vector3d m_center;
    double m_radius;
    double m_height;
    bool m_is_finite;
  };

inline double Cylinder::GetRadius() const 
  {
  return m_radius; 
  }

inline Vector3d Cylinder::GetCenter() const 
  { 
  return m_center; 
  }

inline double Cylinder::GetHeight() const
  { 
  return m_height; 
  }

inline bool Cylinder::IsFinite() const 
  { 
  return m_is_finite; 
  }

inline void Cylinder::SetFiniteness(bool i_is_finite) 
  { 
  m_is_finite = i_is_finite; 
  m_height = -1; 
  };

inline std::string Cylinder::Serialize() const
  {
  return std::string();
  }