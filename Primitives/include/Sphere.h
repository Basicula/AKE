#pragma once

#include <Vector.h>
#include <IObject.h>

class Sphere : public IObject
  {
  public:
    Sphere(const ColorMaterial& i_material = defaultMaterial);
    Sphere(double i_radius, const ColorMaterial& i_material = defaultMaterial);
    Sphere(const Vector3d& i_center, double i_radius, const ColorMaterial& i_material = defaultMaterial);

    inline Vector3d GetCenter() const { return m_center; };
    inline double GetRadius() const { return m_radius; };

    bool GetNormalInPoint(Vector3d& o_normal, const Vector3d& i_point) const override;
  private:
    Vector3d m_center;
    double m_radius;
  };