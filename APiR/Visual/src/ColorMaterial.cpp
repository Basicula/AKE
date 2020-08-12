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
  std::shared_ptr<ILight> i_light) const
  {
  if (!i_light->GetState())
    return m_color * m_ambient;
  const auto& light_direction = i_light->GetDirection(i_point);
  const double diffuse_coef = std::max(0.0, -i_normal.Dot(light_direction));
  const auto light_reflected_direction = ReflectedDirection(i_normal, -light_direction);
  const double specular_coef = pow(std::max(0.0, light_reflected_direction.Dot(i_view_direction)), m_shinines);
  const double light_intensity = i_light->GetIntensityAtPoint(i_point);
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

Color ColorMaterial::GetResultColor(
  const Vector3d& i_normal, 
  const Vector3d& i_light, 
  const Vector3d& i_view) const
  {
  double diffuse = std::max(0.0, i_normal.Dot(i_light));
  double specular = pow(std::max(0.0, (i_normal * i_normal.Dot(i_light) * 2 - i_light).Dot(i_view)), m_shinines);
  double red_factor = m_ambient[0] + m_diffuse[0] * diffuse + m_specular[0] * specular;
  double green_factor = m_ambient[1] + m_diffuse[1] * diffuse + m_specular[1] * specular;
  double blue_factor = m_ambient[2] + m_diffuse[2] * diffuse + m_specular[2] * specular;
  return m_color * Vector3d(red_factor, green_factor, blue_factor);
  }