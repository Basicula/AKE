#include "WorldController.h"

WorldController::WorldController(World* ip_scene)
  : mp_scene(ip_scene)
{}

void WorldController::_ProcessEvent(const Event& i_event)
{
  // Convert window pixel position to scene coords
  if (i_event.Type() == Event::EventType::MOUSE_BUTTON_PRESSED_EVENT)
    m_mouse_pressed_position = Vector2d{ m_mouse_position.first - mp_scene->m_screen_origin[0],
                                         mp_scene->m_screen_origin[1] - m_mouse_position.second };
  if (i_event.Type() == Event::EventType::MOUSE_BUTTON_RELEASED_EVENT) {
    const Vector2d mouse_released_position(m_mouse_position.first - mp_scene->m_screen_origin[0],
                                           mp_scene->m_screen_origin[1] - m_mouse_position.second);
    mp_scene->SpawnBall(mouse_released_position, m_mouse_pressed_position - mouse_released_position);
  }
}
