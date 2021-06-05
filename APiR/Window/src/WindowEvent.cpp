#include <Window/WindowEvent.h>

WindowResizeEvent::WindowResizeEvent(const int i_width, const int i_height)
  : Event(EventType::WINDOW_RESIZED_EVENT)
  , m_width(i_width)
  , m_heigth(i_height) {
}

std::pair<int, int> WindowResizeEvent::Size() const {
  return std::pair<int, int>(m_width, m_heigth);
}

WindowCloseEvent::WindowCloseEvent()
  : Event(EventType::WINDOW_CLOSED_EVENT) {
}