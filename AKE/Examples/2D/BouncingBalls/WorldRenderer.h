#pragma once
#include "Rendering/IRenderer.h"
#include "World.h"

class WorldRenderer : public IRenderer
{
public:
  explicit WorldRenderer(World* ip_scene);

  void Render() override;

private:
  void _OnWindowResize(int i_new_width, int i_new_height) override;

private:
  World* mp_scene;
};
