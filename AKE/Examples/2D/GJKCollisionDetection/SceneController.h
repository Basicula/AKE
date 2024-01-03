#pragma once
#include "Window/EventListner.h"
#include "World.2D/Object2D.h"
#include "World.2D/Scene2D.h"

class SceneController : public EventListner
{
public:
  SceneController(Scene2D& i_scene, std::size_t i_objects_cnt, std::size_t i_window_width, std::size_t i_window_height);

  void PollEvents() override {}

private:
  void _ProcessEvent(const Event& i_event) override;

private:
  Scene2D& m_scene;
  Object2D* mp_focused_object;
  std::size_t m_objects_cnt;
  std::size_t m_window_width;
  std::size_t m_window_height;
};