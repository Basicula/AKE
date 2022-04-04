#pragma once
#include <Geometry.2D/Circle.h>
#include <Rendering.2D/Drawer.h>
#include <Visual/Color.h>
#include <memory>

class CircleDrawer : public Drawer
{
public:
  CircleDrawer(std::shared_ptr<Circle> ip_circle, const Color& i_color, bool i_fill = true);

  void Draw() const override;

private:
  std::shared_ptr<Circle> mp_circle;
  Color m_color;
  bool m_fill;
};