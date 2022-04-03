#include "Rendering/SimpleCamera.h"

#include "Math/Constants.h"

SimpleCamera::SimpleCamera(
  const Vector3d& i_location,
  const Vector3d& i_direction,
  const Vector3d& i_up_vector,
  const double i_fov,
  const double i_aspect)
  : Camera(i_location,
    i_direction,
    i_up_vector)
  , m_fov(i_fov)
  , m_aspect(i_aspect) {
  m_view_angle = m_fov * PI / 180;
  m_half_height = tan(m_view_angle / 2);
  m_half_width = m_half_height * m_aspect;
}

Ray SimpleCamera::CameraRay(const double i_u, const double i_v) const {
  return { m_location, (m_right * i_u * m_half_width * 2 + m_up * i_v * m_half_height * 2 - m_corner).Normalized() };
}

void SimpleCamera::_Update() {
  m_right = m_up.CrossProduct(m_direction).Normalized();
  m_corner = m_right * m_half_width + m_up * m_half_height - m_direction;
}