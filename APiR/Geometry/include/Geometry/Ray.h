#pragma once
#include "Math/Vector.h"

class Ray
{
public:
  Ray() = delete;
  Ray(const Ray& i_other);
  HOSTDEVICE Ray(const Vector3d& i_origin, const Vector3d& i_dir);

  const Vector3d& GetOrigin() const;
  void SetOrigin(const Vector3d& i_origin);
  
  const Vector3d& GetDirection() const;
  void SetDirection(const Vector3d& i_direction);

  Vector3d GetPoint(double i_distance) const;

private:
  Vector3d m_origin;
  Vector3d m_direction;
};
