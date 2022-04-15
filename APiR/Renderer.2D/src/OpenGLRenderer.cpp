#include "Renderer.2D/OpenGLRenderer.h"

#include <GL/glew.h>

OpenGLRenderer::OpenGLRenderer(const int i_width, const int i_height, const Scene2D& i_scene)
  : m_scene(i_scene)
{
  _OnWindowResize(i_width, i_height);
}

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
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, i_new_width, i_new_height, 0.0);
}