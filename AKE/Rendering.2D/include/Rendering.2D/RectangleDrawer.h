#pragma once
#include "Geometry.2D/Rectangle.h"
#include "Rendering.2D/SimpleDrawer.h"
#include "Visual/Color.h"

class RectangleDrawer : public SimpleDrawer
{
public:
  RectangleDrawer(const Rectangle& i_rectangle, const Color& i_color, bool i_fill = true);

  void Draw() const override;

private:
  const Rectangle& m_rectangle;
};