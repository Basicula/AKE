#pragma once
#include <Geometry.2D/Circle.h>
#include <Rendering.2D/SimpleDrawer.h>
#include <Visual/Color.h>

class CircleDrawer : public SimpleDrawer
{
public:
  CircleDrawer(const Circle& i_circle, const Color& i_color, bool i_fill = true);

  void Draw() const override;

private:
  const Circle& m_circle;
};