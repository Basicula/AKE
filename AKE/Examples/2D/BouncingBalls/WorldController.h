#pragma once
#include "Window/EventListner.h"
#include "World.h"

class WorldController : public EventListner
{
public:
  explicit WorldController(World* ip_scene);

  void PollEvents() override {}

private:
  void _ProcessEvent(const Event& i_event) override;

private:
  World* mp_scene;
  Vector2d m_mouse_pressed_position;
};