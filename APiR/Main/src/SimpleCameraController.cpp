#include <Main/SimpleCameraController.h>

SimpleCameraController::SimpleCameraController(Camera* ip_camera)
  : mp_camera(ip_camera)
  , m_movement_step(0.1) {}

void SimpleCameraController::ProcessEvent(const Event& i_event) {
  switch (i_event.Type()) {
    case Event::EventType::KEY_PRESSED_EVENT:
    {
      const auto& keyboard_event = static_cast<const KeyPressedEvent&>(i_event);
      const auto& camera_position = mp_camera->GetLocation();
      const auto& camera_direction = mp_camera->GetDirection();
      const auto& camera_right = mp_camera->GetRight();
      const auto& camera_up = mp_camera->GetUpVector();
      KeyboardButton key = keyboard_event.Key();
      if (key == KeyboardButton::KEY_W)
        mp_camera->SetLocation(camera_position + camera_direction * m_movement_step);
      else if (key == KeyboardButton::KEY_S)
        mp_camera->SetLocation(camera_position - camera_direction * m_movement_step);
      else if (key == KeyboardButton::KEY_A)
        mp_camera->SetLocation(camera_position - camera_right * m_movement_step);
      else if (key == KeyboardButton::KEY_D)
        mp_camera->SetLocation(camera_position + camera_right * m_movement_step);
      else if (key == KeyboardButton::KEY_X)
        mp_camera->SetLocation(camera_position - camera_up * m_movement_step);
      else if (key == KeyboardButton::KEY_Z)
        mp_camera->SetLocation(camera_position + camera_up * m_movement_step);
      return;
    }
    default: return;
  }
}