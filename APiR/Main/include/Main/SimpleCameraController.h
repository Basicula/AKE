#pragma once
#include <Math/Vector.h>
#include <Rendering/Camera.h>
#include <Window/EventListner.h>

class SimpleCameraController : public EventListner {
public:
  SimpleCameraController(Camera* ip_camera);

  virtual void PollEvents() override;

private:
  virtual void _ProcessEvent(const Event& i_event) override;

  void _MoveCamera();

  void _RotateCamera();

private:
  Camera* mp_camera;
  double m_movement_step;
  Vector2d m_prev_mouse_position;
  bool m_rotation_enabled;
};