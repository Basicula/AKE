#pragma once
#include <Geometry.2D/Circle.h>
#include <Rendering.2D/Drawer.h>
#include <Visual/Color.h>

class CircleDrawer : public Drawer
{
public:
  CircleDrawer(const Circle& i_circle, const Color& i_color, bool i_fill = true);

  void Draw() const override;

private:
  const Circle& m_circle;
  Color m_color;
  bool m_fill;
};