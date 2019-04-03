#include "Ray.h"

Ray::Ray(const Vector3d& i_start, const Vector3d& i_dir, double i_environment)
  : m_start(i_start)
  , m_direction(i_dir.Normalized())
  , m_environment(i_environment)
  {}

Ray::Ray(const Ray& i_other)
  : m_start(i_other.m_start)
  , m_direction(i_other.m_direction)
  , m_environment(i_other.m_environment)
  {}