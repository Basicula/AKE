#include "Window/MouseEvent.h"

MouseButtonPressedEvent::MouseButtonPressedEvent(const MouseButton i_button)
  : Event(EventType::MOUSE_BUTTON_PRESSED_EVENT)
  , m_button(i_button)
{}

MouseButton MouseButtonPressedEvent::Button() const
{
  return m_button;
}

MouseButtonReleasedEvent::MouseButtonReleasedEvent(const MouseButton i_button)
  : Event(EventType::MOUSE_BUTTON_RELEASED_EVENT)
  , m_button(i_button)
{}

MouseButton MouseButtonReleasedEvent::Button() const
{
  return m_button;
}

MouseMovedEvent::MouseMovedEvent(const double i_x, const double i_y)
  : Event(EventType::MOUSE_MOVED_EVENT)
  , m_x(i_x)
  , m_y(i_y)
{}

std::pair<double, double> MouseMovedEvent::Position() const
{
  return { m_x, m_y };
}

MouseScrollEvent::MouseScrollEvent(const double i_offset)
  : Event(EventType::MOUSE_SCOLLED_EVENT)
  , m_offset(i_offset)
{}

double MouseScrollEvent::Offset() const
{
  return m_offset;
}