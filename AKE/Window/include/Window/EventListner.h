#pragma once
#include "Window/Event.h"
#include "Window/Keys.h"

#include <utility>

class EventListner
{
public:
  EventListner();
  virtual ~EventListner() = default;

  void ProcessEvent(const Event& i_event);
  virtual void PollEvents() = 0;

protected:
  virtual void _ProcessEvent(const Event& i_event) = 0;

  [[nodiscard]] bool _IsKeyPressed(KeyboardButton i_key) const;

protected:
  static constexpr std::size_t mg_key_count = 256;
  bool m_key_pressed[mg_key_count];
  std::pair<double, double> m_mouse_position;
};
