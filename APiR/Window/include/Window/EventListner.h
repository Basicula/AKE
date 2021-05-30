#pragma once
#include <Window/Event.h>

class EventListner {
public:
  virtual ~EventListner() = default;

  virtual void ProcessEvent(const Event& i_event) = 0;
};