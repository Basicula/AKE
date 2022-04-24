#pragma once
#include "Window/EventListner.h"
#include "TowerDefense.h"

class GameController : public EventListner
{
public:
  explicit GameController(TowerDefense* ip_game);

  void PollEvents() override {}

private:
  void _ProcessEvent(const Event& i_event) override;

private:
  TowerDefense* mp_game;
};