#pragma once
#include <Math/Vector.h>

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

private:
  Vector3d m_origin;
  Vector3d m_direction;
};

inline const Vector3d& Ray::GetOrigin() const 
  { 
  return m_origin; 
  };
  
inline void Ray::SetOrigin(const Vector3d& i_origin)
  { 
  m_origin = i_origin; 
  };
  
inline const Vector3d& Ray::GetDirection() const 
  { 
  return m_direction; 
  };
  
inline void Ray::SetDirection(const Vector3d& i_direction)
  { 
  m_direction = i_direction; 
  };
