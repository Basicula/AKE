#include <Visual/Camera.h>
#include <Math/Constants.h>

#include <cmath>

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
  const double theta = m_fov * PI / 180;
  const double halfHeight = std::tan(theta / 2);
  const double halfWidth = halfHeight * m_aspect;

  m_right = m_up.CrossProduct(m_direction).Normalized();
  m_corner = m_right * halfWidth + m_up * halfHeight - m_direction * m_focusDistance;
  m_u = m_right * halfWidth * 2;
  m_v = m_up * halfHeight * 2;
  }