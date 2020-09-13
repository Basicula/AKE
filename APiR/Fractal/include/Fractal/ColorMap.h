#pragma once
#include <Visual/Color.h>

#include <vector>

// maps size_t from interval [from, to] to Color from m_colors
class ColorMap
  {
  public:
    virtual ~ColorMap() = default;

    virtual Color operator()(
      std::size_t i_val, 
      std::size_t i_max_val) const = 0;

  protected:
    ColorMap(const std::vector<Color>& i_colors);

  protected:
    std::vector<Color> m_colors;
  };