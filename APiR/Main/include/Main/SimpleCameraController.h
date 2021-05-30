#pragma once
#include <Rendering/Camera.h>
#include <Window/EventListner.h>

class SimpleCameraController : public EventListner {
public:
  SimpleCameraController(Camera* ip_camera);

  virtual void ProcessEvent(const Event& i_event) override;
private:
  Camera* mp_camera;
  double m_movement_step;
};