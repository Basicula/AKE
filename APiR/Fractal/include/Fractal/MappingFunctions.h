#pragma once
#include "Macros.h"
#include "Memory/custom_vector.h"
#include "Visual/Color.h"

namespace FractalMapping {
  HOSTDEVICE Color Default(size_t i_val, const custom_vector<Color>& i_colors);
  HOSTDEVICE Color Smooth(size_t i_val, size_t i_max_val, const custom_vector<Color>& i_colors);
}