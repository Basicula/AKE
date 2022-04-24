#include "WorldRenderer.h"

#include <GL/glew.h>

WorldRenderer::WorldRenderer(World* ip_scene)
  : mp_scene(ip_scene)
{
  const auto size = mp_scene->GetSize();
  _OnWindowResize(static_cast<int>(size.first), static_cast<int>(size.second));
}

void WorldRenderer::Render()
{
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  for (const auto& p_object : mp_scene->GetObjects())
    p_object->GetDrawer().Draw();
}

void WorldRenderer::_OnWindowResize(const int i_new_width, const int i_new_height)
{
  glViewport(0, 0, i_new_width, i_new_height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-i_new_width / 2, i_new_width / 2, -i_new_height / 2, i_new_height / 2, 0.0, 1.0);
}
