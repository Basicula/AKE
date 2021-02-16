#pragma once
#include <Macro/CudaMacro.h>

#include <CUDACore/HostDeviceBuffer.h>

#include <Visual/Color.h>

class Image
  {
  public:
    Image(
      std::size_t i_width, 
      std::size_t i_height, 
      std::uint32_t i_default_color = 0);
    ~Image();

    std::uint32_t GetPixel(std::size_t i_x, std::size_t i_y) const;
    void GetPixelRGBA(
      std::size_t i_x, 
      std::size_t i_y,
      std::uint8_t& o_red,
      std::uint8_t& o_green,
      std::uint8_t& o_blue,
      std::uint8_t& o_alpha) const;
    HOSTDEVICE void SetPixel(
      std::size_t i_x, 
      std::size_t i_y, 
      std::uint32_t i_color);
    void SetPixelRGBA(
      std::size_t i_x, 
      std::size_t i_y, 
      std::uint8_t i_red,
      std::uint8_t i_green,
      std::uint8_t i_blue,
      std::uint8_t i_alpha = 255);

    HOSTDEVICE std::size_t GetWidth() const;
    HOSTDEVICE std::size_t GetHeight() const;
    std::size_t GetDepth() const;
    std::size_t GetSize() const;
    std::size_t GetBytesCount() const;

    std::uint32_t* GetData() const;
    std::uint8_t* GetRGBAData() const;

  private:
    HOSTDEVICE std::size_t _ID(std::size_t i_x, std::size_t i_y) const;

  private:
    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_size;
    HostDeviceBuffer<std::uint32_t> mp_pixels;

    static const unsigned char m_bytes_per_pixel = 4;
  };

inline std::size_t Image::_ID(std::size_t i_x, std::size_t i_y) const
  {
  return i_y * m_width + i_x;
  }

inline std::uint32_t Image::GetPixel(std::size_t i_x, std::size_t i_y) const
  {
  return mp_pixels[_ID(i_x, i_y)];
  }

inline void Image::GetPixelRGBA(
  std::size_t i_x, 
  std::size_t i_y, 
  std::uint8_t& o_red, 
  std::uint8_t& o_green, 
  std::uint8_t& o_blue, 
  std::uint8_t& o_alpha) const
  {
  auto rgba_data = reinterpret_cast<const std::uint8_t*>(&mp_pixels[_ID(i_x, i_y)]);
  o_red   = rgba_data[0];
  o_green = rgba_data[1];
  o_blue  = rgba_data[2];
  o_alpha = rgba_data[3];
  }

inline void Image::SetPixel(
  std::size_t i_x, 
  std::size_t i_y, 
  std::uint32_t i_color)
  {
  mp_pixels[_ID(i_x, i_y)] = i_color;
  }

inline std::size_t Image::GetWidth() const
  {
  return m_width;
  }

inline std::size_t Image::GetHeight() const
  {
  return m_height;
  }

inline std::size_t Image::GetSize() const
  {
  return m_size;
  }

inline std::size_t Image::GetBytesCount() const
  {
  return GetSize() * GetDepth();
  }
  
inline std::size_t Image::GetDepth() const
  {
  return m_bytes_per_pixel;
  }

inline std::uint32_t* Image::GetData() const
  {
  return mp_pixels.data();
  }