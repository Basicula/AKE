#include "Rendering.2D/CircleDrawer.h"

#include "Math/Constants.h"

#include <GL/glew.h>

CircleDrawer::CircleDrawer(const Circle& i_circle, const Color& i_color, const bool i_fill)
  : m_circle(i_circle)
  , m_color(i_color)
  , m_fill(i_fill)
{}

void CircleDrawer::Draw() const
{
  const auto& x = m_circle.center[0];
  const auto& y = m_circle.center[1];
  const auto& radius = m_circle.radius;
  constexpr auto segments_count = 180;
  if (m_fill) {
    glBegin(GL_TRIANGLE_FAN);
    glVertex2d(x, y);
  } else
    glBegin(GL_LINE_LOOP);
  glColor3ub(m_color.GetRed(), m_color.GetGreen(), m_color.GetBlue());
  for (std::size_t segment_id = 0; segment_id <= segments_count; segment_id++) {
    const auto angle = static_cast<double>(segment_id) * Math::Constants::TWOPI / segments_count;
    glVertex2d(x + radius * cos(angle), y + radius * sin(angle));
  }
  glEnd();
}
