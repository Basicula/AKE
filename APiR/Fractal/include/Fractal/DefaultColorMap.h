#pragma once
#include "Fractal/ColorMap.h"

class DefaultColorMap : public ColorMap
  {
  public:
    DefaultColorMap(const std::vector<Color>& i_colors);

    virtual Color operator()(
      std::size_t i_val,
      std::size_t i_max_val) const override;
  };