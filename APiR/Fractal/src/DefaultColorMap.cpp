#include <Fractal/DefaultColorMap.h>

DefaultColorMap::DefaultColorMap(const HostDeviceBuffer<Color>& i_colors)
  : ColorMap(i_colors)
  {
  }

Color DefaultColorMap::operator()(
  std::size_t i_val,
  std::size_t /*i_max_val*/) const
  {
  return m_colors[i_val % m_colors.size()];
  }
