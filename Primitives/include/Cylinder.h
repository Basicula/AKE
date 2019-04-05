#pragma once

#include <IObject.h>
#include <Vector.h>

class Cylinder : public IObject
  {
  public:
    Cylinder() = delete;
    //if i_height < 0 let cylinder be infinite
    //-1 default value for infiniteness
    Cylinder(const Vector3d& i_center, double i_radius, double i_height = -1, const ColorMaterial& i_material = g_DefaultMaterial);

    inline double GetRadius() const { return m_radius; };
    inline Vector3d GetCenter() const { return m_center; };
    inline double GetHeight() const { return m_height; };
    inline bool IsFinite() const { return m_is_finite; };

    inline void SetFiniteness(bool i_is_finite) { m_is_finite = i_is_finite; m_height = -1; };

    bool GetNormalInPoint(Vector3d& o_normal, const Vector3d& i_point) const override;
  private:
    Vector3d m_center;
    double m_radius;
    double m_height;
    bool m_is_finite;
  };
