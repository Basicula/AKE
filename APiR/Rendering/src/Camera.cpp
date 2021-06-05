#include <Rendering/Camera.h>
#include <Math/Constants.h>
#include <Geometry/Transformation.h>

Camera::Camera(
    const Vector3d& i_lookFrom
  , const Vector3d& i_lookAt
  , const Vector3d& i_up
  , double i_fov
  , double i_aspect
  , double i_focusDist)
  : m_location(i_lookFrom)
  , m_direction((i_lookAt - i_lookFrom).Normalized())
  , m_up(i_up.Normalized())
  , m_fov(i_fov)
  , m_aspect(i_aspect)
  , m_focusDistance(i_focusDist)
  {
  _Init();
  }

void Camera::_Init() {
  const double theta = m_fov * PI / 180;
  const double halfHeight = tan(theta / 2);
  const double halfWidth = halfHeight * m_aspect;

  m_right = m_up.CrossProduct(m_direction).Normalized();
  m_corner = m_right * halfWidth + m_up * halfHeight - m_direction * m_focusDistance;
  m_u = m_right * halfWidth * 2;
  m_v = m_up * halfHeight * 2;
}

const Vector3d& Camera::GetDirection() const {
  return m_direction;
}

const Vector3d& Camera::GetUpVector() const {
  return m_up;
}

const Vector3d& Camera::GetRight() const {
  return m_right;
}

void Camera::Move(const Vector3d& i_displacement_vector) {
  m_location += i_displacement_vector;
}

void Camera::Rotate(const Vector3d& i_rotation_axis, const double i_angle_in_rad) {
  Transformation rotation;
  rotation.SetRotation(i_rotation_axis, i_angle_in_rad);
  rotation.Rotate(m_direction);
  rotation.Rotate(m_up);
  _Init();
}
