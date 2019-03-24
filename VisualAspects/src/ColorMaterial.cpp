#include <vector>
#include <memory>

#include <ColorMaterial.h>

ColorMaterial::ColorMaterial(const Color& i_color, const Vector3d& i_ambient, const Vector3d& i_diffuse, const Vector3d& i_specular, double i_shinines, double i_reflection, double i_refraction)
  : m_color(i_color)
  , m_shinines(i_shinines)
  , m_ambient(i_ambient)
  , m_diffuse(i_diffuse)
  , m_specular(i_specular)
  , m_reflection(i_reflection)
  , m_refraction(i_refraction)
  {
  }

Color ColorMaterial::GetResultColor(const Vector3d& i_normal, const Vector3d& i_light, const Vector3d& i_view) const
  {
  double diffuse = std::max(0.0, i_normal.Dot(i_light));
  double specular = pow(std::max(0.0, (i_normal * i_normal.Dot(i_light) * 2 - i_light).Dot(i_view)), m_shinines);
  double red_factor = m_ambient[0] + m_diffuse[0] * diffuse + m_specular[0] * specular;
  double green_factor = m_ambient[1] + m_diffuse[1] * diffuse + m_specular[1] * specular;
  double blue_factor = m_ambient[2] + m_diffuse[2] * diffuse + m_specular[2] * specular;
  return m_color * Vector3d(red_factor, green_factor, blue_factor);
  }

Color ColorMaterial::GetResultColor(const Vector3d& i_normal, const Vector3d& i_point, const std::vector<SpotLight*>& i_lights, const Vector3d& i_view) const
  {
  double la = 0.2;
  double red_factor = m_ambient[0] * la;
  double green_factor = m_ambient[1] * la;
  double blue_factor = m_ambient[2] * la;
  for (const auto& light : i_lights)
    {
    Vector3d to_light = (light->GetLocation() - i_point).Normalized();
    double diffuse = std::max(0.01, i_normal.Dot(to_light));
    double specular = pow(std::max(0.0, (i_normal * i_normal.Dot(to_light) * 2 - to_light).Dot(i_view)), m_shinines);
    red_factor    += m_ambient[0] * la + (light->GetIntensityAtPoint(i_point)) * (m_diffuse[0] * diffuse + m_specular[0] * specular);
    green_factor  += m_ambient[1] * la + (light->GetIntensityAtPoint(i_point)) * (m_diffuse[1] * diffuse + m_specular[1] * specular);
    blue_factor   += m_ambient[2] * la + (light->GetIntensityAtPoint(i_point)) * (m_diffuse[2] * diffuse + m_specular[2] * specular);
    }
  return m_color * Vector3d(red_factor, green_factor, blue_factor);
  }