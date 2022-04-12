#include "Rendering.2D/CircleDrawer.h"

#include "Math/Constants.h"

#include <GL/glew.h>

CircleDrawer::CircleDrawer(const Circle& i_circle, const Color& i_color, const bool i_fill)
  : SimpleDrawer(i_color, i_fill)
  , m_circle(i_circle)
{}

void CircleDrawer::Draw() const
{
  const auto& radius = m_circle.m_radius;
  constexpr auto segments_count = 180;
  const auto x = mp_transformation != nullptr ? mp_transformation->GetTranslation()[0] : 0.0;
  const auto y = mp_transformation != nullptr ? mp_transformation->GetTranslation()[1] : 0.0;
  glColor3ub(m_color.GetRed(), m_color.GetGreen(), m_color.GetBlue());
  if (m_fill) {
    glBegin(GL_TRIANGLE_FAN);
    glVertex2d(x, y);
  } else
    glBegin(GL_LINE_LOOP);
  for (std::size_t segment_id = 0; segment_id <= segments_count; segment_id++) {
    const auto angle = static_cast<double>(segment_id) * Math::Constants::TWOPI / segments_count;
    glVertex2d(x + radius * cos(angle), y + radius * sin(angle));
  }
  glEnd();
}
