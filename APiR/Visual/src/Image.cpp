#include <Image.h>

Image::Image(
  std::size_t i_width, 
  std::size_t i_height, 
  std::uint32_t i_default_color)
  : m_width(i_width)
  , m_height(i_height)
  , m_size(i_width * i_height)
  , mp_pixels(new std::uint32_t[m_size])
  , mp_rgb_data(new std::uint8_t[m_size * m_bytes_per_pixel])
  {
  std::fill_n(mp_pixels, m_size, i_default_color);
  }

Image::~Image()
  {
  if (mp_pixels)
    delete[] mp_pixels;
  mp_pixels = nullptr;
  }

std::uint8_t* Image::GetRGBData() const
  {
  for (std::size_t i = 0; i < m_size; ++i)
    {
    mp_rgb_data[i * m_bytes_per_pixel + 0] = static_cast<std::uint8_t>((mp_pixels[i] & 0xff0000) >> 16);
    mp_rgb_data[i * m_bytes_per_pixel + 1] = static_cast<std::uint8_t>((mp_pixels[i] & 0x00ff00) >> 8 );
    mp_rgb_data[i * m_bytes_per_pixel + 2] = static_cast<std::uint8_t>((mp_pixels[i] & 0x0000ff) >> 0 );
    }
  return mp_rgb_data;
  }