#pragma once
#include "Math/Vector.h"
#include "Visual/Color.h"
#include "Visual/ILight.h"

class SpotLight final : public ILight
{
public:
  explicit SpotLight(const Vector3d& i_location,
                     const Color& i_color = Color(255, 255, 255),
                     double i_intensity = 1.0,
                     bool i_state = true);

  [[nodiscard]] Vector3d GetLocation() const;
  void SetLocation(const Vector3d& i_location);

  [[nodiscard]] Color GetColor() const;
  void SetColor(const Color& i_color);

  [[nodiscard]] Vector3d GetDirection(const Vector3d& i_point) const override;

  [[nodiscard]] double GetIntensityAtPoint(const Vector3d& i_point) const override;

  [[nodiscard]] std::string Serialize() const override;

private:
  Vector3d m_location;
  Color m_color;
};
