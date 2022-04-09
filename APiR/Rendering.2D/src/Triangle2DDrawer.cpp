#include "Rendering.2D/Triangle2DDrawer.h"

#include <GL/glew.h>

Triangle2DDrawer::Triangle2DDrawer(const Triangle2D& i_triangle, const Color& i_color, bool i_fill)
  : m_triangle(i_triangle)
  , m_color(i_color)
  , m_fill(i_fill)
{}

void Triangle2DDrawer::Draw() const
{
  auto vertices = m_triangle.m_vertices;
  if (mp_transformation != nullptr)
    for (auto& vertex : vertices)
      mp_transformation->Transform(vertex);

  if (m_fill)
    glBegin(GL_TRIANGLES);
  else
    glBegin(GL_LINE_LOOP);
  glColor3ub(m_color.GetRed(), m_color.GetGreen(), m_color.GetBlue());
  for (const auto& vertex : vertices)
    glVertex2d(vertex[0], vertex[1]);
  glEnd();
}
