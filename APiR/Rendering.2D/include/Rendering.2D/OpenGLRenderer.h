#pragma once
#include "Rendering/IRenderer.h"
#include "Rendering.2D/Scene2D.h"

class OpenGLRenderer : public IRenderer
{
public:
  explicit OpenGLRenderer(const Scene2D& i_scene);

  void Render() override;

private:
  void _OnWindowResize(int i_new_width, int i_new_height) override;

private:
  const Scene2D& m_scene;
};