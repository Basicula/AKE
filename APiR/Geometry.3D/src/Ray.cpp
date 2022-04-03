#include <Geometry.3D/Ray.h>

Ray::Ray(const Ray& i_other)
  : m_origin(i_other.m_origin)
  , m_direction(i_other.m_direction) {
  }

Ray::Ray(const Vector3d& i_origin, const Vector3d& i_dir)
  : m_origin(i_origin)
  , m_direction(i_dir) {
  }

const Vector3d& Ray::GetOrigin() const {
  return m_origin;
  };

void Ray::SetOrigin(const Vector3d& i_origin) {
  m_origin = i_origin;
  };

const Vector3d& Ray::GetDirection() const {
  return m_direction;
  };

void Ray::SetDirection(const Vector3d& i_direction) {
  m_direction = i_direction;
  };

Vector3d Ray::GetPoint(double i_distance) const {
  return m_origin + m_direction * i_distance;
  }