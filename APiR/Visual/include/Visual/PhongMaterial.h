#pragma once
#include "Math/Vector.h"
#include "Visual/Color.h"
#include "Visual/IVisualMaterial.h"

class PhongMaterial final : public IVisualMaterial
{
public:
  explicit PhongMaterial(const Color& i_color = Color(0xffaaaaaa),
                         const Vector3d& i_ambient = Vector3d(1.0, 1.0, 1.0),
                         const Vector3d& i_diffuse = Vector3d(1.0, 1.0, 1.0),
                         const Vector3d& i_specular = Vector3d(1.0, 1.0, 1.0),
                         double i_shinines = 1.0,
                         double i_reflection = 0.0,
                         double i_refraction = 0.0);

  [[nodiscard]] Color GetPrimitiveColor() const override;

  [[nodiscard]] Color CalculateColor(const Vector3d& i_normal,
                                             const Vector3d& i_view_direction,
                                             const Vector3d& i_light_direction) const override;

  [[nodiscard]] bool IsReflectable() const override;
  [[nodiscard]] Vector3d ReflectedDirection(const Vector3d& i_normal_at_point,
                                                    const Vector3d& i_view_direction) const override;
  [[nodiscard]] double ReflectionInfluence() const override;

  [[nodiscard]] bool IsRefractable() const override;
  [[nodiscard]] Vector3d RefractedDirection() const override;
  [[nodiscard]] double RefractionInfluence() const override;

  [[nodiscard]] Color GetAmbientColor() const;
  [[nodiscard]] Color GetDiffuseColor() const;
  [[nodiscard]] Color GetSpecularColor() const;

private:
  Color m_color;
  double m_shinines;

  // x,y,z == r,g,b(coefs) in [0,1]
  Vector3d m_ambient;
  Vector3d m_diffuse;
  Vector3d m_specular;

  // TODO remove
  //[0,1]
  double m_reflection;
  double m_refraction;
};
