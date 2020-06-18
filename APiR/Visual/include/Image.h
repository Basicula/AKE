#pragma once
#include <vector>
#include <string>


class Image
  {
  public:
    Image(
      std::size_t i_width, 
      std::size_t i_height, 
      std::uint32_t i_default_color = 0x000000);
    ~Image();

    std::uint32_t GetPixel(std::size_t i_x, std::size_t i_y) const;
    void SetPixel(std::size_t i_x, std::size_t i_y, std::uint32_t i_color);

    std::size_t GetWidth() const;
    std::size_t GetHeight() const;
    std::size_t GetSize() const;
    std::size_t GetDepth() const;

    std::uint32_t* GetData() const;
    std::uint8_t* GetRGBData() const;

  private:
    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_size;
    std::uint32_t* mp_pixels;
    std::uint8_t* mp_rgb_data;

    static const unsigned char m_bytes_per_pixel = 3;
  };

inline std::uint32_t Image::GetPixel(std::size_t i_x, std::size_t i_y) const
  {
  // TODO maybe change to throw
  if(i_x >= m_width || i_y >= m_height)
    return 0;
  return mp_pixels[i_y * m_width + i_x];
  }

inline void Image::SetPixel(
  std::size_t i_x, 
  std::size_t i_y, 
  std::uint32_t i_color)
  {
  if (i_x >= m_width || i_y >= m_height)
    return;
  mp_pixels[i_y * m_width + i_x] = i_color;
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