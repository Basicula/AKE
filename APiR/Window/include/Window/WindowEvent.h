#pragma once
#include <Window/Event.h>

#include <utility>

class WindowResizeEvent : public Event {
public:
  WindowResizeEvent(const int i_width, const int i_height);

  std::pair<int, int> Size() const;

private:
  int m_width;
  int m_heigth;
};

class WindowCloseEvent : public Event {
public:
  WindowCloseEvent();
};