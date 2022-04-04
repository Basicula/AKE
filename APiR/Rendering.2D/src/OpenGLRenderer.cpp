#include "Rendering.2D/OpenGLRenderer.h"

OpenGLRenderer::OpenGLRenderer(const Scene2D& i_scene)
  : m_scene(i_scene)
{}

void OpenGLRenderer::Render() const
{
  const auto& objects = m_scene.GetObjects();
  for (const auto& p_object : objects)
    p_object->mp_drawer->Draw();
}
