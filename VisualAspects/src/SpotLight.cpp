#include <SpotLight.h>

SpotLight::SpotLight(const Vector3d& i_location, const Color& i_color, double i_intensity)
  : m_location(i_location)
  , m_color(i_color)
  , m_intensity(i_intensity)
  {};

double SpotLight::GetIntensityAtPoint(const Vector3d& i_point) const
  {
  double distance_to_point = m_location.Distance(i_point);
  double c1 = 0.0;
  double c2 = 0.1;
  double c3 = 0.0;
  return m_intensity / (c1 + c2 * distance_to_point + c3 * distance_to_point * distance_to_point);
  }