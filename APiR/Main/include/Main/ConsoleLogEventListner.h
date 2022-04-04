#pragma once
#include "Window/EventListner.h"

class ConsoleLogEventListner : public EventListner
{
public:
  virtual void PollEvents() override;

protected:
  virtual void _ProcessEvent(const Event& i_event) override;
};