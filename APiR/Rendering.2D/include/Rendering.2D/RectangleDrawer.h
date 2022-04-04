#pragma once
#include "Geometry.2D/Rectangle.h"
#include "Rendering.2D/Drawer.h"
#include "Visual/Color.h"

#include <memory>

class RectangleDrawer : public Drawer
{
public:
  RectangleDrawer(std::shared_ptr<Rectangle> ip_rectangle, const Color& i_color, bool i_fill = true);

  void Draw() const override;

private:
  std::shared_ptr<Rectangle> mp_rectangle;
  Color m_color;
  bool m_fill;
};