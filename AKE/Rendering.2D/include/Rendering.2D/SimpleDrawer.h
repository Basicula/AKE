#pragma once
#include "Rendering.2D/Drawer.h"
#include "Visual/Color.h"

struct SimpleDrawer : public Drawer {
    Color m_color;
    bool m_fill;

    SimpleDrawer(const Color& i_color, bool i_fill);
  };
