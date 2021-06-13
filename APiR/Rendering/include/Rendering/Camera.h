#pragma once
#include <Geometry/Ray.h>

#include <Math/Vector.h>

class Camera
  {
  public:
    Camera(
      const Vector3d& i_location,
      const Vector3d& i_direction,
      const Vector3d& i_up_vector);
    virtual ~Camera() = default;

    HOSTDEVICE const Vector3d& GetLocation() const;
    void SetLocation(const Vector3d& i_location);

    const Vector3d& GetDirection() const;
    const Vector3d& GetUpVector() const;
    const Vector3d& GetRight() const;

    // Move camera location at some vector
    // i.e. camera_location + i_displacement_vector
    void Move(const Vector3d& i_displacement_vector);
    void Rotate(const Vector3d& i_rotation_axis, const double i_angle_in_rad);

    HOSTDEVICE virtual Ray CameraRay(const double i_u, const double i_v) const = 0;

  protected:
    // Updates derivated class specific data after some camera base manipulations
    virtual void _Update() = 0;

  protected:
    Vector3d m_location;
    Vector3d m_direction;
    Vector3d m_up;
    Vector3d m_right;
  };