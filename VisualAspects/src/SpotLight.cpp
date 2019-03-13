#include <SpotLight.h>

SpotLight::SpotLight(const Vector3d& i_location, const Color& i_color)
  : m_location(i_location)
  , m_color(i_color)
  , m_intensity(100)
  {};

Vector3d SpotLight::GetLocation() const
  {
  return m_location;
  }

Color SpotLight::GetColor() const
  {
  return m_color;
  }
