#include "Rendering.2D/CircleDrawer.h"

#include "Math/Constants.h"

#include <GL/glew.h>

CircleDrawer::CircleDrawer(std::shared_ptr<Circle> ip_circle, const Color& i_color, const bool i_fill)
  : mp_circle(std::move(ip_circle))
  , m_color(i_color)
  , m_fill(i_fill)
{}

void CircleDrawer::Draw() const
{
  const auto& x = mp_circle->center[0];
  const auto& y = mp_circle->center[1];
  const auto& radius = mp_circle->radius;
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
