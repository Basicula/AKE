#include <Fractal/SmoothColorMap.h>

#include <cmath>

SmoothColorMap::SmoothColorMap()
  : ColorMap({})
  {
  }

SmoothColorMap::SmoothColorMap(const std::vector<Color>& i_colors)
  : ColorMap(i_colors)
  {
  }

Color SmoothColorMap::operator()(
  std::size_t i_val,
  std::size_t i_max_val) const
  {
  if (m_colors.empty())
    return Color();
  double scaled = 1.0 * (m_colors.size() - 1) * i_val / i_max_val;
  std::size_t nearby_id = static_cast<std::size_t>(scaled);
  return 
    m_colors[nearby_id] * (scaled - nearby_id) + 
    m_colors[(nearby_id + 1) % m_colors.size()] * (nearby_id + 1 - scaled);
  }