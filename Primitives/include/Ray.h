#pragma once

#include <Vector.h>

class Ray
{
public:
  Ray() = delete;
  Ray(const Vector3d& i_start, const Vector3d& i_dir);

  inline Vector3d GetStart() const { return m_start; };
  inline Vector3d GetDirection() const { return m_direction; };
private:
  Vector3d m_start;
  Vector3d m_direction;
};
