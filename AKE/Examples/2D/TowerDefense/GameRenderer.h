#pragma once
#include "Rendering/IRenderer.h"
#include "TowerDefense.h"

class GameRenderer : public IRenderer
{
public:
  explicit GameRenderer(TowerDefense* ip_game);

  void Render() override;

private:
  void _OnWindowResize(const int i_new_width, const int i_new_height) override;

private:
  TowerDefense* mp_game;
};