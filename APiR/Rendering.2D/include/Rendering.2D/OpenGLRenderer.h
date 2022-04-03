#pragma once
#include "Rendering.2D/Scene2D.h"

class OpenGLRenderer
{
public:
  explicit OpenGLRenderer(const Scene2D& i_scene);

  void Render() const;

private:
  const Scene2D& m_scene;
};