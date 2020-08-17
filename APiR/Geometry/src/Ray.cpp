#include <Geometry/Ray.h>

Ray::Ray(const Ray& i_other)
  : m_origin(i_other.m_origin)
  , m_direction(i_other.m_direction)
  {}

Ray::Ray(const Vector3d& i_origin, const Vector3d& i_dir)
  : m_origin(i_origin)
  , m_direction(i_dir)
  {}