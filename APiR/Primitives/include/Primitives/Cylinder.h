#pragma once
#include <Primitives/ISurface.h>
#include <Primitives/Plane.h>

#include <Math/Vector.h>

#include <optional>

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
    void SetCenter(const Vector3d& i_center);
    
    double GetHeight() const;

    bool IsFinite() const;
    void SetFiniteness(bool i_is_finite);

    virtual std::string Serialize() const override;
  protected:
    virtual void _CalculateBoundingBox() override;
    virtual Vector3d _NormalAtLocalPoint(const Vector3d& i_local_point) const override;
    virtual bool _IntersectWithRay(
      double& io_nearest_intersection_dist,
      const Ray& i_ray) const override;
  private:
    double m_radius;
    double m_height;
    bool m_is_finite;

    // helpfull vars
    double m_zmax;
    double m_zmin;
    double m_half_height;

    std::optional<Plane> m_top;
    std::optional<Plane> m_bottom;
  };

inline double Cylinder::GetRadius() const 
  {
  return m_radius; 
  }

inline Vector3d Cylinder::GetCenter() const 
  { 
  return GetTranslation(); 
  }

inline void Cylinder::SetCenter(const Vector3d& i_center)
  {
  SetTranslation(i_center);
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