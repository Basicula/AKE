#pragma once
#include "Macros.h"
#ifdef __CUDA_ARCH__
#include <cuda_runtime.h>
#else
#include <algorithm>
#endif

namespace Utils {
  template<class T>
  HOSTDEVICE const T& min(const T& i_first, const T& i_second) {
#ifdef __CUDA_ARCH__
    return i_first < i_second ? i_first : i_second;
#else
    return std::min(i_first, i_second);
#endif
    }

  template<class T>
  HOSTDEVICE const T& max(const T& i_first, const T& i_second) {
#ifdef __CUDA_ARCH__
    return i_first > i_second ? i_first : i_second;
#else
    return std::max(i_first, i_second);
#endif
    }
  }