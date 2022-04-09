#pragma once
#include "Geometry.2D/Triangle2D.h"
#include "Rendering.2D/Drawer.h"
#include "Visual/Color.h"

class Triangle2DDrawer : public Drawer
{
public:
  Triangle2DDrawer(const Triangle2D& i_triangle, const Color& i_color, bool i_fill = true);

  void Draw() const override;

private:
  const Triangle2D& m_triangle;
  Color m_color;
  bool m_fill;
};
