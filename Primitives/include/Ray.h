#pragma once

#include <Vector.h>

class Ray
{
public:
  Ray() = delete;
  Ray(const Vector3d& i_start, const Vector3d& i_dir, double i_environment = 1.0);
  Ray(const Ray& i_other);

  inline Vector3d GetStart() const { return m_start; };
  inline Vector3d GetDirection() const { return m_direction; };
  inline double GetEnvironment() const { return m_environment; };
  inline void SetEnvironment(double i_environment) { m_environment = i_environment; };

private:
  Vector3d m_start;
  Vector3d m_direction;
  double m_environment;
};
