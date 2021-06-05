#include <Window/KeyboardEvent.h>

KeyPressedEvent::KeyPressedEvent(const KeyboardButton i_key)
  : Event(EventType::KEY_PRESSED_EVENT)
  , m_key(i_key) {
}

KeyboardButton KeyPressedEvent::Key() const {
  return m_key;
}

KeyReleasedEvent::KeyReleasedEvent(const KeyboardButton i_key)
  : Event(EventType::KEY_RELEASED_EVENT)
  , m_key(i_key) {
}

KeyboardButton KeyReleasedEvent::Key() const {
  return m_key;
}