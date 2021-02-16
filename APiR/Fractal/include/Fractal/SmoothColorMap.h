#pragma once
#include <Fractal/ColorMap.h>

class SmoothColorMap : public ColorMap
  {
  public:
    SmoothColorMap();
    SmoothColorMap(const HostDeviceBuffer<Color>& i_colors);

    void AddColor(const Color& i_color);

    HOSTDEVICE virtual Color operator()(
      std::size_t i_val,
      std::size_t i_max_val) const;
  };

inline void SmoothColorMap::AddColor(const Color& i_color)
  {
  m_colors.push_back(i_color);
  }