#include "Renderer.2D/OpenGLRenderer.h"

#include <GL/glew.h>

OpenGLRenderer::OpenGLRenderer(const Scene2D& i_scene)
  : m_scene(i_scene)
{}

void OpenGLRenderer::Render()
{
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  const auto& objects = m_scene.GetObjects();
  for (const auto& p_object : objects)
    p_object->GetDrawer().Draw();
}

void OpenGLRenderer::_OnWindowResize(const int i_new_width, const int i_new_height)
{
  glViewport(0, 0, i_new_width, i_new_height);
}