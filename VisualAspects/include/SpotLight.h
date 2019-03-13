#pragma once

#include <Vector.h>
#include <Color.h>

class SpotLight
{
public:
  SpotLight(const Vector3d& i_location, const Color& i_color = Color(255,255,255));

  Vector3d GetLocation() const;
  Color GetColor() const;

private:
  Vector3d m_location;
  Color m_color;
  double m_intensity;
};