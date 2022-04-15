#pragma once
#include "Macros.h"
#include "Memory/MemoryManager.h"
#include "Visual/Color.h"

class Image
{
public:
  Image(std::size_t i_width, std::size_t i_height, std::uint32_t i_default_color = 0);
  ~Image();

  [[nodiscard]] std::uint32_t GetPixel(std::size_t i_x, std::size_t i_y) const;
  void GetPixelRGBA(std::size_t i_x,
                    std::size_t i_y,
                    uint8_t& o_red,
                    uint8_t& o_green,
                    uint8_t& o_blue,
                    uint8_t& o_alpha) const;
  HOSTDEVICE void SetPixel(std::size_t i_x, std::size_t i_y, std::uint32_t i_color);
  void SetPixelRGBA(std::size_t i_x,
                    std::size_t i_y,
                    uint8_t i_red,
                    uint8_t i_green,
                    uint8_t i_blue,
                    uint8_t i_alpha = 255);

  HOSTDEVICE [[nodiscard]] std::size_t GetWidth() const;
  HOSTDEVICE [[nodiscard]] std::size_t GetHeight() const;
  [[nodiscard]] std::size_t GetDepth() const;
  [[nodiscard]] std::size_t GetSize() const;
  [[nodiscard]] std::size_t GetBytesCount() const;

  [[nodiscard]] std::uint32_t* GetData() const;
  [[nodiscard]] uint8_t* GetRGBAData() const;

private:
  HOSTDEVICE [[nodiscard]] std::size_t _ID(std::size_t i_x, std::size_t i_y) const;

private:
  std::size_t m_width;
  std::size_t m_height;
  std::size_t m_size;
  MemoryManager::pointer<std::uint32_t> m_pixels;

  static constexpr unsigned char m_bytes_per_pixel = 4;
};
