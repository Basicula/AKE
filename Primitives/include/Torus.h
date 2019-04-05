#pragma once

#include <Vector.h>
#include <IObject.h>
#include <ColorMaterial.h>

//based on torus equation
/*
z^2 = a^2 - (b - sqrt(x^2 + y^2))^2
i.e.
z^2 = m_minor_radius - (m_major_radius - sqrt(x^2+y^2))^2
*/
class Torus : public IObject
  {
  public:
    Torus() = delete;
    Torus(const Vector3d& i_center, double i_major_radius, double i_minor_radius, const ColorMaterial& i_material = g_DefaultMaterial);

    inline Vector3d GetCenter() const { return m_center; };
    inline double GetMinor() const { return m_minor_radius; };
    inline double GetMajor() const { return m_major_radius; };

    bool GetNormalInPoint(Vector3d& o_normal, const Vector3d& i_point) const override;

  private:
    Vector3d m_center;
    double m_major_radius;
    double m_minor_radius;
  };