#include "GameController.h"
#include "Window/MouseEvent.h"

GameController::GameController(TowerDefense* ip_game)
  : mp_game(ip_game)
{}

void GameController::_ProcessEvent(const Event& i_event)
{
  if (i_event.Type() == Event::EventType::MOUSE_BUTTON_PRESSED_EVENT) {
    const auto& mouse_button_pressed_event = static_cast<const MouseButtonPressedEvent&>(i_event);
    if (mouse_button_pressed_event.Button() == MouseButton::MOUSE_LEFT_BUTTON) {
      auto shoot_direction = Vector2d{ m_mouse_position.first, m_mouse_position.second } -
                             mp_game->m_player.GetTransformation().GetTranslation();
      shoot_direction.Normalize();
      mp_game->Shoot(shoot_direction);
    }
  }
  if (i_event.Type() == Event::EventType::MOUSE_MOVED_EVENT) {
    const auto& mouse_moved_event = static_cast<const MouseMovedEvent&>(i_event);
    const auto& position = mouse_moved_event.Position();
    m_mouse_position = { position.first, position.second };
  }
}