#include <Window/Event.h>

Event::Event(const EventType i_type)
  : m_type(i_type) {}

Event::EventType Event::Type() const {
  return m_type;
}

KeyPressedEvent::KeyPressedEvent(const KeyboardButton i_key)
  : Event(EventType::KEY_PRESSED_EVENT)
  , m_key(i_key) {}

KeyboardButton KeyPressedEvent::Key() const {
  return m_key;
}

MouseButtonPressedEvent::MouseButtonPressedEvent(const MouseButton i_button)
  : Event(EventType::MOUSE_BUTTON_PRESSED_EVENT)
  , m_button(i_button) {}

MouseButton MouseButtonPressedEvent::Button() const {
  return m_button;
}

MouseButtonReleasedEvent::MouseButtonReleasedEvent(const MouseButton i_button)
  : Event(EventType::MOUSE_BUTTON_RELEASED_EVENT)
  , m_button(i_button) {}

MouseButton MouseButtonReleasedEvent::Button() const {
  return m_button;
}

MouseMovedEvent::MouseMovedEvent(const double i_x, const double i_y) 
  : Event(EventType::MOUSE_MOVED_EVENT)
  , m_x(i_x)
  , m_y(i_y) {}

std::pair<double, double> MouseMovedEvent::Position() const {
  return std::pair<double, double>(m_x, m_y);
}

KeyReleasedEvent::KeyReleasedEvent(const KeyboardButton i_key)
  : Event(EventType::KEY_RELEASED_EVENT)
  , m_key(i_key) {}

KeyboardButton KeyReleasedEvent::Key() const {
  return m_key;
}

MouseScrollEvent::MouseScrollEvent(const double i_offset) 
  : Event(EventType::MOUSE_SCOLLED_EVENT)
  , m_offset(i_offset) {}

double MouseScrollEvent::Offset() const {
  return m_offset;
}

KeyRepeatEvent::KeyRepeatEvent(const KeyboardButton i_key)
  : Event(EventType::KEY_REPEAT_EVENT)
  , m_key(i_key) {}

KeyboardButton KeyRepeatEvent::Key() const {
  return m_key;
}

WindowResizeEvent::WindowResizeEvent(const int i_width, const int i_height) 
  : Event(EventType::WINDOW_RESIZED_EVENT)
  , m_width(i_width)
  , m_heigth(i_height) {}

std::pair<int, int> WindowResizeEvent::Size() const {
  return std::pair<int, int>(m_width, m_heigth);
}

WindowCloseEvent::WindowCloseEvent()
  : Event(EventType::WINDOW_CLOSED_EVENT) {}
