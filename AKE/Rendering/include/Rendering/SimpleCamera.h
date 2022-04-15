#pragma once
#include "Rendering/Camera.h"

class SimpleCamera : public Camera
{
public:
  // fov in degrees
  // aspect = width / height
  SimpleCamera(const Vector3d& i_location,
               const Vector3d& i_direction,
               const Vector3d& i_up_vector,
               const double i_fov,
               const double i_aspect);

  HOSTDEVICE virtual Ray CameraRay(const double i_u, const double i_v) const override;

protected:
  virtual void _Update() override;

private:
  Vector3d m_corner;

  double m_fov;
  double m_aspect;

  double m_view_angle;
  double m_half_height;
  double m_half_width;
};