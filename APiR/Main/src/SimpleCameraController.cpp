#include "Main/SimpleCameraController.h"

#include "Geometry/Transformation.h"
#include "Math/Constants.h"
#include "Window/MouseEvent.h"

SimpleCameraController::SimpleCameraController(Camera* ip_camera)
  : EventListner()
  , mp_camera(ip_camera)
  , m_movement_step(1)
  , m_angular_speed(Math::Constants::PI / 45)
  , m_rotation_enabled(false)
{}

void SimpleCameraController::PollEvents()
{
  _MoveCamera();
}

void SimpleCameraController::_ProcessEvent(const Event& i_event)
{
  const auto event_type = i_event.Type();
  if (event_type == Event::EventType::MOUSE_MOVED_EVENT)
    _RotateCamera();
  else if (event_type == Event::EventType::MOUSE_BUTTON_PRESSED_EVENT) {
    const auto& mouse_event = static_cast<const MouseButtonPressedEvent&>(i_event);
    if (mouse_event.Button() == MouseButton::MOUSE_LEFT_BUTTON)
      m_rotation_enabled = true;
  } else if (event_type == Event::EventType::MOUSE_BUTTON_RELEASED_EVENT) {
    m_rotation_enabled = false;
  }
}

void SimpleCameraController::_MoveCamera()
{
  Vector3d direction;
  if (_IsKeyPressed(KeyboardButton::KEY_W))
    direction += mp_camera->GetDirection();
  if (_IsKeyPressed(KeyboardButton::KEY_S))
    direction += -mp_camera->GetDirection();
  if (_IsKeyPressed(KeyboardButton::KEY_A))
    direction += -mp_camera->GetRight();
  if (_IsKeyPressed(KeyboardButton::KEY_D))
    direction += mp_camera->GetRight();
  if (_IsKeyPressed(KeyboardButton::KEY_X))
    direction += -mp_camera->GetUpVector();
  if (_IsKeyPressed(KeyboardButton::KEY_Z))
    direction += mp_camera->GetUpVector();
  mp_camera->Move(direction * m_movement_step);
}

void SimpleCameraController::_RotateCamera()
{
  if (m_rotation_enabled) {
    const auto dx = m_mouse_position.first - m_prev_mouse_position[0];
    const auto dy = m_mouse_position.second - m_prev_mouse_position[1];
    const auto angle = atan2(dy, dx);
    const auto& camera_direction = mp_camera->GetDirection();
    Transformation3D rotation;
    rotation.SetRotation(camera_direction, angle);
    auto new_camera_right = mp_camera->GetRight();
    rotation.Rotate(new_camera_right);
    const auto rotation_axis = new_camera_right.CrossProduct(camera_direction);
    mp_camera->Rotate(rotation_axis, m_angular_speed);
  }
  m_prev_mouse_position[0] = m_mouse_position.first;
  m_prev_mouse_position[1] = m_mouse_position.second;
}
