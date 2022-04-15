#include "Geometry.2D/Rectangle.h"

Rectangle::Rectangle(const double i_width, const double i_height)
  : m_width(i_width)
  , m_height(i_height)
  , m_corners({ Vector2d{ -i_width / 2, -i_height / 2 },
                Vector2d{ +i_width / 2, -i_height / 2 },
                Vector2d{ +i_width / 2, +i_height / 2 },
                Vector2d{ -i_width / 2, +i_height / 2 } })
{}

void Rectangle::_CalculateBoundingBox()
{
  m_bounding_box = BoundingBox2D{ m_corners[0], m_corners[2] };
}
