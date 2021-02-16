#pragma once
#include <CUDACore/HostDeviceBuffer.h>
#include <Visual/Color.h>

#include <vector>

// maps size_t from interval [from, to] to Color from m_colors
class ColorMap
  {
  public:
    virtual ~ColorMap() = default;

    HOSTDEVICE virtual Color operator()(
      std::size_t i_val, 
      std::size_t i_max_val) const = 0;

  protected:
    HOSTDEVICE ColorMap(const HostDeviceBuffer<Color>& i_colors);

  protected:
    HostDeviceBuffer<Color> m_colors;
  };