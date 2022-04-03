#include "Visual/SpotLight.h"

SpotLight::SpotLight(const Vector3d& i_location, const Color& i_color, const double i_intensity, const bool i_state)
  : ILight(i_state, i_intensity)
  , m_location(i_location)
  , m_color(i_color){};

double SpotLight::GetIntensityAtPoint(const Vector3d& i_point) const
{
  if (!m_state)
    return 0;
  const double distance_to_point = m_location.Distance(i_point);
  constexpr double c1 = 0.0;
  constexpr double c2 = 0.05;
  constexpr double c3 = 0.0;
  return m_intensity / (c1 + c2 * distance_to_point + c3 * distance_to_point * distance_to_point);
}

Vector3d SpotLight::GetLocation() const
{
  return m_location;
};

void SpotLight::SetLocation(const Vector3d& i_location)
{
  m_location = i_location;
};

Color SpotLight::GetColor() const
{
  return m_color;
};

void SpotLight::SetColor(const Color& i_color)
{
  m_color = i_color;
};

std::string SpotLight::Serialize() const
{
  std::string res = "{ \"SpotLight\" : { ";
  res += " \"Location\" : " + m_location.Serialize() + ", ";
  res += " \"Color\" : " + m_color.Serialize() + ", ";
  res += " \"Intensity\" : " + std::to_string(m_intensity) + ", ";
  res += " \"State\" : " + std::to_string(m_state);
  res += " } }";
  return res;
}

Vector3d SpotLight::GetDirection(const Vector3d& i_point) const
{
  return (i_point - m_location).Normalized();
}