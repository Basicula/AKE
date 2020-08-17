#include <Rendering/Image.h>

Image::Image(
  std::size_t i_width, 
  std::size_t i_height, 
  std::uint32_t i_default_color)
  : m_width(i_width)
  , m_height(i_height)
  , m_size(i_width * i_height)
  , mp_pixels(new std::uint32_t[m_size])
  {
  std::fill_n(mp_pixels, m_size, i_default_color);
  }

Image::~Image()
  {
  free(mp_pixels);
  }

void Image::SetPixelRGBA(
  std::size_t i_x, 
  std::size_t i_y, 
  std::uint8_t i_red, 
  std::uint8_t i_green, 
  std::uint8_t i_blue, 
  std::uint8_t i_alpha)
  {
  auto rgba_data = reinterpret_cast<std::uint8_t*>(mp_pixels + _ID(i_x, i_y));
  rgba_data[0] = i_red;
  rgba_data[1] = i_green;
  rgba_data[2] = i_blue;
  rgba_data[3] = i_alpha;
  }

std::uint8_t* Image::GetRGBAData() const
  {
  return reinterpret_cast<std::uint8_t*>(mp_pixels);
  }