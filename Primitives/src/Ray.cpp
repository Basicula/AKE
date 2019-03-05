#include "Ray.h"

Ray::Ray(const Vector3d& i_start, const Vector3d& i_dir)
  : m_start(i_start)
  , m_direction(i_dir.Normalized())
{}