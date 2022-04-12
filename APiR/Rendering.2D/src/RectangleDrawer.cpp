#include "Rendering.2D/RectangleDrawer.h"

#include <GL/glew.h>
#include <array>

RectangleDrawer::RectangleDrawer(const Rectangle& i_rectangle, const Color& i_color, const bool i_fill)
  : SimpleDrawer(i_color, i_fill)
  , m_rectangle(i_rectangle)
{}

void RectangleDrawer::Draw() const
{
  glColor3ub(m_color.GetRed(), m_color.GetGreen(), m_color.GetBlue());
  if (m_fill)
    glBegin(GL_QUADS);
  else
    glBegin(GL_LINE_LOOP);
  for (auto corner : m_rectangle.m_corners) {
    if (mp_transformation)
      mp_transformation->Transform(corner);
    glVertex2d(corner[0], corner[1]);
  }
  glEnd();
}
