#pragma once
#include "Window/Event.h"

#include <utility>

class WindowResizeEvent final : public Event
{
public:
  WindowResizeEvent(int i_width, int i_height);

  [[nodiscard]] std::pair<int, int> Size() const;

private:
  int m_width;
  int m_heigth;
};

class WindowCloseEvent final : public Event
{
public:
  WindowCloseEvent();
};