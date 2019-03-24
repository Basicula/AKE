#pragma once

#include <Vector.h>
#include <Color.h>

class SpotLight
  {
  public:
    SpotLight(const Vector3d& i_location, const Color& i_color = Color(255, 255, 255), double i_intensity = 1.0);

    inline Vector3d GetLocation() const { return m_location; };
    inline Color GetColor() const { return m_color; };
    inline double GetIntensity() const { return m_intensity; };
    double GetIntensityAtPoint(const Vector3d& i_point) const;

  private:
    Vector3d m_location;
    Color m_color;
    double m_intensity;
  };