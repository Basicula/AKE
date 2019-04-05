#pragma once

#include <IObject.h>
#include <Vector.h>

class Cylinder : public IObject
  {
  public:
    Cylinder() = delete;
    Cylinder(const Vector3d& i_center, double i_radius, double i_height, const ColorMaterial& i_material = g_DefaultMaterial);

    inline double GetRadius() const { return m_radius; };
    inline Vector3d GetCenter() const { return m_center; };
    inline double GetHeight() const { return m_height; };

    bool GetNormalInPoint(Vector3d& o_normal, const Vector3d& i_point) const override;
  private:
    Vector3d m_center;
    double m_radius;
    double m_height;
  };
