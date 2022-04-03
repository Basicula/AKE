#include "Window/EventListner.h"

#include <algorithm>

EventListner::EventListner()
  : m_key_pressed()
{
  std::fill_n(m_key_pressed, mg_key_count, false);
}

void EventListner::ProcessEvent(const Event& i_event)
{
  switch (i_event.Type()) {
    case Event::EventType::KEY_PRESSED_EVENT:
      m_key_pressed[static_cast<unsigned int>(static_cast<const KeyPressedEvent&>(i_event).Key())] = true;
      break;
    case Event::EventType::KEY_RELEASED_EVENT:
      m_key_pressed[static_cast<unsigned int>(static_cast<const KeyReleasedEvent&>(i_event).Key())] = false;
      break;
    case Event::EventType::MOUSE_MOVED_EVENT:
      m_mouse_position = static_cast<const MouseMovedEvent&>(i_event).Position();
      break;
    default:
      break;
  }
  _ProcessEvent(i_event);
}

bool EventListner::_IsKeyPressed(const KeyboardButton i_key) const
{
  return m_key_pressed[static_cast<unsigned int>(i_key)];
}
