#include "Rendering/Camera.h"

#include "Geometry/Transformation.h"
#include "Math/Constants.h"

Camera::Camera(const Vector3d& i_location, const Vector3d& i_direction, const Vector3d& i_up_vector)
  : m_location(i_location)
  , m_direction(i_direction.Normalized())
  , m_up(i_up_vector.Normalized())
{}

const Vector3d& Camera::GetDirection() const
{
  return m_direction;
}

const Vector3d& Camera::GetUpVector() const
{
  return m_up;
}

const Vector3d& Camera::GetRight() const
{
  return m_right;
}

void Camera::Move(const Vector3d& i_displacement_vector)
{
  m_location += i_displacement_vector;
  _Update();
}

void Camera::Rotate(const Vector3d& i_rotation_axis, const double i_angle_in_rad)
{
  Transformation3D rotation;
  rotation.SetRotation(i_rotation_axis, i_angle_in_rad);
  rotation.RotateOnly(m_direction);
  rotation.RotateOnly(m_up);
  _Update();
}

const Vector3d& Camera::GetLocation() const
{
  return m_location;
}

void Camera::SetLocation(const Vector3d& i_location)
{
  m_location = i_location;
}
