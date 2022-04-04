#include "Rendering.2D/RectangleDrawer.h"

#include <GL/glew.h>

RectangleDrawer::RectangleDrawer(std::shared_ptr<Rectangle> ip_rectangle, const Color& i_color, const bool i_fill)
  : mp_rectangle(std::move(ip_rectangle))
  , m_color(i_color)
  , m_fill(i_fill)
{}

void RectangleDrawer::Draw() const
{
  const double half_width = mp_rectangle->m_width / 2;
  const double half_height = mp_rectangle->m_height / 2;
  glColor3ub(m_color.GetRed(), m_color.GetGreen(), m_color.GetBlue());
  if (m_fill)
    glBegin(GL_QUADS);
  else
    glBegin(GL_LINE_LOOP);
  glVertex2d(mp_rectangle->m_position[0] - half_width, mp_rectangle->m_position[1] - half_height);
  glVertex2d(mp_rectangle->m_position[0] + half_width, mp_rectangle->m_position[1] - half_height);
  glVertex2d(mp_rectangle->m_position[0] + half_width, mp_rectangle->m_position[1] + half_height);
  glVertex2d(mp_rectangle->m_position[0] - half_width, mp_rectangle->m_position[1] + half_height);
  glEnd();
}
