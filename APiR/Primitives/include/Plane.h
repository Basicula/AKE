#pragma once
#include <DefinesAndConstants.h>
#include <Vector.h>
#include <ISurface.h>

class Plane : public ISurface
  {
  public:
    Plane() = delete;
    Plane(
      const Vector3d& i_first, 
      const Vector3d& i_second, 
      const Vector3d& i_third);
    Plane(const Vector3d& i_point, const Vector3d& i_normal);

    Vector3d GetNormal() const;

    double GetValueFromEquation(const Vector3d& i_point) const;

    virtual std::string Serialize() const override;
  protected:
    virtual BoundingBox _GetBoundingBox() const override;
    virtual bool _IntersectWithRay(
      IntersectionRecord& o_intersection,
      const Ray& i_ray) const override;
    virtual Vector3d _NormalAtPoint(const Vector3d& i_point) const override;

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

inline Vector3d Plane::_NormalAtPoint(const Vector3d& /*i_point*/) const
  {
  return m_normal;
  }

inline BoundingBox Plane::_GetBoundingBox() const
  {
  BoundingBox res;
  res.AddPoint(m_point + m_normal * 0.01);
  res.AddPoint(m_point - m_normal * 0.01);
  Vector3d vec1(m_normal[2], m_normal[0], m_normal[1]);
  Vector3d vec2(m_normal[1], m_normal[2], m_normal[0]);
  res.AddPoint(m_point + vec1 * 100);
  res.AddPoint(m_point - vec1 * 100);
  res.AddPoint(m_point + vec2 * 100);
  res.AddPoint(m_point - vec2 * 100);
  return res;
  }

inline std::string Plane::Serialize() const
  {
  return std::string();
  }