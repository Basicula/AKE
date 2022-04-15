#pragma once
#include <limits>

namespace Common::Constants {
  // 1.79769e+308
  constexpr double MAX_DOUBLE = std::numeric_limits<double>::max();

  // 2.22507e-308
  constexpr double MIN_DOUBLE = std::numeric_limits<double>::min();

  // 2147483647
  constexpr int MAX_INT = std::numeric_limits<int>::max();

  // -2147483648
  constexpr int MIN_INT = std::numeric_limits<int>::min();
}