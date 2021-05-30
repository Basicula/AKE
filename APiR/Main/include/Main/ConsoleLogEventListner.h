#pragma once
#include <Window/EventListner.h>

class ConsoleLogEventListner : public EventListner {
public:
  virtual void ProcessEvent(const Event& i_event) override;
};