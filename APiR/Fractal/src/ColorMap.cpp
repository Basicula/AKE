#include <Fractal/ColorMap.h>

HOSTDEVICE ColorMap::ColorMap(const HostDeviceBuffer<Color>& i_colors)
  : m_colors(i_colors)
  {
  }
