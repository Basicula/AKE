#include "Visual/PhongMaterial.h"

#include "Math/VectorOperations.h"

PhongMaterial::PhongMaterial(const Color& i_color,
                             const Vector3d& i_ambient,
                             const Vector3d& i_diffuse,
                             const Vector3d& i_specular,
                             const double i_shinines,
                             const double i_reflection,
                             const double i_refraction)
  : m_color(i_color)
  , m_shinines(i_shinines)
  , m_ambient(i_ambient)
  , m_diffuse(i_diffuse)
  , m_specular(i_specular)
  , m_reflection(i_reflection)
  , m_refraction(i_refraction)
{}

Color PhongMaterial::CalculateColor(const Vector3d& i_normal,
                                    const Vector3d& i_view_direction,
                                    const Vector3d& i_light_direction) const
{
  const double diffuse_coef = std::max(0.0, -i_normal.Dot(i_light_direction));
  const auto light_reflected_direction = Math::Reflected(i_normal, -i_light_direction);
  const double specular_coef = pow(std::max(0.0, light_reflected_direction.Dot(i_view_direction)), m_shinines);
  return m_color * (m_ambient + m_diffuse * diffuse_coef + m_specular * specular_coef);
}

Vector3d PhongMaterial::ReflectedDirection(const Vector3d& i_normal_at_point, const Vector3d& i_view_direction) const
{
  return Math::Reflected(i_normal_at_point, i_view_direction);
}

Vector3d PhongMaterial::RefractedDirection() const
{
  return {};
}

Color PhongMaterial::GetAmbientColor() const
{
  return m_color * m_ambient;
};

Color PhongMaterial::GetDiffuseColor() const
{
  return m_color * m_diffuse;
};

Color PhongMaterial::GetSpecularColor() const
{
  return m_color * m_specular;
};

bool PhongMaterial::IsReflectable() const
{
  return m_reflection > 0.0;
}

double PhongMaterial::ReflectionInfluence() const
{
  return m_reflection;
}

bool PhongMaterial::IsRefractable() const
{
  return m_refraction > 0.0;
}

double PhongMaterial::RefractionInfluence() const
{
  return m_refraction;
}

Color PhongMaterial::GetPrimitiveColor() const
{
  return GetAmbientColor();
}