#pragma once
#include <Macros.h>

#include <Memory/MemoryManager.h>
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
      uint8_t& o_red,
      uint8_t& o_green,
      uint8_t& o_blue,
      uint8_t& o_alpha) const;
    HOSTDEVICE void SetPixel(
      std::size_t i_x,
      std::size_t i_y,
      std::uint32_t i_color);
    void SetPixelRGBA(
      std::size_t i_x,
      std::size_t i_y,
      uint8_t i_red,
      uint8_t i_green,
      uint8_t i_blue,
      uint8_t i_alpha = 255);

    HOSTDEVICE std::size_t GetWidth() const;
    HOSTDEVICE std::size_t GetHeight() const;
    std::size_t GetDepth() const;
    std::size_t GetSize() const;
    std::size_t GetBytesCount() const;

    std::uint32_t* GetData() const;
    uint8_t* GetRGBAData() const;

  private:
    HOSTDEVICE std::size_t _ID(std::size_t i_x, std::size_t i_y) const;

  private:
    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_size;
    MemoryManager::pointer<std::uint32_t> m_pixels;

    static const unsigned char m_bytes_per_pixel = 4;
  };

inline std::size_t Image::_ID(std::size_t i_x, std::size_t i_y) const {
  return i_y * m_width + i_x;
  }

inline std::uint32_t Image::GetPixel(std::size_t i_x, std::size_t i_y) const {
  return m_pixels.mp_data[_ID(i_x, i_y)];
  }

inline void Image::GetPixelRGBA(
  std::size_t i_x,
  std::size_t i_y,
  uint8_t& o_red,
  uint8_t& o_green,
  uint8_t& o_blue,
  uint8_t& o_alpha) const {
  auto rgba_data = reinterpret_cast<const uint8_t*>(&m_pixels.mp_data[_ID(i_x, i_y)]);
  o_red = rgba_data[0];
  o_green = rgba_data[1];
  o_blue = rgba_data[2];
  o_alpha = rgba_data[3];
  }

inline void Image::SetPixel(
  std::size_t i_x,
  std::size_t i_y,
  std::uint32_t i_color) {
  m_pixels.mp_data[_ID(i_x, i_y)] = i_color;
  }

inline std::size_t Image::GetWidth() const {
  return m_width;
  }

inline std::size_t Image::GetHeight() const {
  return m_height;
  }

inline std::size_t Image::GetSize() const {
  return m_size;
  }

inline std::size_t Image::GetBytesCount() const {
  return GetSize() * GetDepth();
  }

inline std::size_t Image::GetDepth() const {
  return m_bytes_per_pixel;
  }

inline std::uint32_t* Image::GetData() const {
  return m_pixels.mp_data;
  }