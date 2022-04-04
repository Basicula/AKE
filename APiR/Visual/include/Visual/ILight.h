#pragma once
#include "Math/Vector.h"

#include <string>

class ILight
{
public:
  ILight(bool i_state, double i_intensity);
  virtual ~ILight() = default;

  void SetState(bool i_state);
  [[nodiscard]] bool GetState() const;

  void SetIntensity(double i_intensity);
  [[nodiscard]] double GetIntensity() const;
  [[nodiscard]] virtual double GetIntensityAtPoint(const Vector3d& i_point) const = 0;

  [[nodiscard]] virtual Vector3d GetDirection(const Vector3d& i_point) const = 0;

  [[nodiscard]] virtual std::string Serialize() const = 0;

protected:
  bool m_state;
  double m_intensity;
};
