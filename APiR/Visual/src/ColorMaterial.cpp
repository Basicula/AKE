#include <Visual/ColorMaterial.h>

ColorMaterial::ColorMaterial(
  const Color& i_color, 
  const Vector3d& i_ambient, 
  const Vector3d& i_diffuse, 
  const Vector3d& i_specular, 
  double i_shinines, 
  double i_reflection, 
  double i_refraction)
  : m_color(i_color)
  , m_shinines(i_shinines)
  , m_ambient(i_ambient)
  , m_diffuse(i_diffuse)
  , m_specular(i_specular)
  , m_reflection(i_reflection)
  , m_refraction(i_refraction)
  {
  }

Color ColorMaterial::GetLightInfluence(
  const Vector3d& i_point,
  const Vector3d& i_normal,
  const Vector3d& i_view_direction,
  const ILight* ip_light) const
  {
  if (!ip_light->GetState())
    return m_color * m_ambient;
  const auto& light_direction = ip_light->GetDirection(i_point);
  const double diffuse_coef = std::max(0.0, -i_normal.Dot(light_direction));
  const auto light_reflected_direction = ReflectedDirection(i_normal, -light_direction);
  const double specular_coef = pow(std::max(0.0, light_reflected_direction.Dot(i_view_direction)), m_shinines);
  const double light_intensity = ip_light->GetIntensityAtPoint(i_point);
  return m_color * (m_ambient + m_diffuse * diffuse_coef + m_specular * specular_coef) * light_intensity;
  }

Vector3d ColorMaterial::ReflectedDirection(const Vector3d& i_normal_at_point, const Vector3d& i_view_direction) const
  {
  return i_normal_at_point * i_normal_at_point.Dot(-i_view_direction) * 2.0 + i_view_direction;
  }

Vector3d ColorMaterial::RefractedDirection() const
  {
  return Vector3d();
  }