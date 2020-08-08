#pragma once
#include <vector>
#include <string>

#include <Color.h>

class Image
  {
  public:
    Image(
      std::size_t i_width, 
      std::size_t i_height, 
      const Color& i_default_color = 0xffffffff);
    ~Image();

    Color GetPixel(std::size_t i_x, std::size_t i_y) const;
    void SetPixel(
      std::size_t i_x, 
      std::size_t i_y, 
      const Color& i_color);
    void SetPixelRGBA(
      std::size_t i_x, 
      std::size_t i_y, 
      std::uint8_t i_red,
      std::uint8_t i_green,
      std::uint8_t i_blue,
      std::uint8_t i_alpha = 255);

    std::size_t GetWidth() const;
    std::size_t GetHeight() const;
    std::size_t GetSize() const;
    std::size_t GetDepth() const;

    std::uint32_t* GetData() const;
    std::uint8_t* GetRGBAData() const;

  private:
    std::size_t _ID(std::size_t i_x, std::size_t i_y) const;

  private:
    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_size;
    std::uint32_t* mp_pixels;

    static const unsigned char m_bytes_per_pixel = 4;
  };

inline std::size_t Image::_ID(std::size_t i_x, std::size_t i_y) const
  {
  return i_y * m_width + i_x;
  }

inline Color Image::GetPixel(std::size_t i_x, std::size_t i_y) const
  {
  return mp_pixels[_ID(i_x, i_y)];
  }

inline void Image::SetPixel(
  std::size_t i_x, 
  std::size_t i_y, 
  const Color& i_color)
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
  
inline std::size_t Image::GetDepth() const
  {
  return m_bytes_per_pixel;
  }

inline std::uint32_t* Image::GetData() const
  {
  return mp_pixels;
  }