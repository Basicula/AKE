#include "Image/Image.h"

Image::Image(const std::size_t i_width, const std::size_t i_height, std::uint32_t /*i_default_color*/)
  : m_width(i_width)
  , m_height(i_height)
  , m_size(i_width * i_height)
  , m_pixels{ MemoryManager::AllocateType::Undefined, nullptr }
{
  m_pixels = MemoryManager::allocate<uint32_t>(m_size);
}

Image::~Image()
{
  MemoryManager::clean(m_pixels);
}

void Image::SetPixelRGBA(const std::size_t i_x,
                         const std::size_t i_y,
                         const uint8_t i_red,
                         const uint8_t i_green,
                         const uint8_t i_blue,
                         const uint8_t i_alpha)
{
  auto* rgba_data = reinterpret_cast<uint8_t*>(&m_pixels.mp_data[_ID(i_x, i_y)]);
  rgba_data[0] = i_red;
  rgba_data[1] = i_green;
  rgba_data[2] = i_blue;
  rgba_data[3] = i_alpha;
}

uint8_t* Image::GetRGBAData() const
{
  return reinterpret_cast<uint8_t*>(m_pixels.mp_data);
}

std::size_t Image::_ID(const std::size_t i_x, const std::size_t i_y) const
{
  return i_y * m_width + i_x;
}

std::uint32_t Image::GetPixel(const std::size_t i_x, const std::size_t i_y) const
{
  return m_pixels.mp_data[_ID(i_x, i_y)];
}

void Image::GetPixelRGBA(const std::size_t i_x,
                         const std::size_t i_y,
                         uint8_t& o_red,
                         uint8_t& o_green,
                         uint8_t& o_blue,
                         uint8_t& o_alpha) const
{
  auto rgba_data = reinterpret_cast<const uint8_t*>(&m_pixels.mp_data[_ID(i_x, i_y)]);
  o_red = rgba_data[0];
  o_green = rgba_data[1];
  o_blue = rgba_data[2];
  o_alpha = rgba_data[3];
}

void Image::SetPixel(const std::size_t i_x, const std::size_t i_y, const std::uint32_t i_color)
{
  m_pixels.mp_data[_ID(i_x, i_y)] = i_color;
}

std::size_t Image::GetWidth() const
{
  return m_width;
}

std::size_t Image::GetHeight() const
{
  return m_height;
}

std::size_t Image::GetSize() const
{
  return m_size;
}

std::size_t Image::GetBytesCount() const
{
  return GetSize() * GetDepth();
}

std::size_t Image::GetDepth() const
{
  return m_bytes_per_pixel;
}

std::uint32_t* Image::GetData() const
{
  return m_pixels.mp_data;
}
