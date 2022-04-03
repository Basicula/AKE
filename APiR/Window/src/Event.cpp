#include "Window/Event.h"

Event::Event(const EventType i_type)
  : m_type(i_type) {}

Event::EventType Event::Type() const {
  return m_type;
}
