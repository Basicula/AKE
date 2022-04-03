#include "Fractal/MappingFunctions.h"

namespace FractalMapping {
  Color Default(size_t i_val, const custom_vector<Color>& i_colors) {
    if (i_colors.empty())
      return Color(11, 22, 33);
    return i_colors[i_val % i_colors.size()];
    }

  Color Smooth(size_t i_val, size_t i_max_val, const custom_vector<Color>& i_colors) {
    if (i_colors.empty())
      return Color(11,22,33);
    double scaled = 1.0 * ( i_colors.size() - 1 ) * i_val / i_max_val;
    size_t nearby_id = static_cast<size_t>( scaled );
    return
      i_colors[nearby_id] * ( scaled - nearby_id ) +
      i_colors[( nearby_id + 1 ) % i_colors.size()] * ( nearby_id + 1 - scaled );
    }
  }