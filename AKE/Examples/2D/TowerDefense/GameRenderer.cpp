#include "GameRenderer.h"

#include <GL/glew.h>

GameRenderer::GameRenderer(TowerDefense* ip_game)
  : mp_game(ip_game)
{
  _OnWindowResize(static_cast<int>(mp_game->m_width), static_cast<int>(mp_game->m_height));
}

void GameRenderer::Render()
{
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  mp_game->m_player.GetDrawer().Draw();
  for (const auto& p_bullet : mp_game->m_bullets)
    p_bullet->GetDrawer().Draw();
  for (const auto& p_mob : mp_game->m_mobs)
    p_mob->GetDrawer().Draw();
}

void GameRenderer::_OnWindowResize(const int i_new_width, const int i_new_height)
{
  glViewport(0, 0, i_new_width, i_new_height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, i_new_width, i_new_height, 0.0, 0.0, 1.0);
  // glOrtho(-i_new_width, i_new_width, i_new_height, -i_new_height, 0.0, 1.0);
}