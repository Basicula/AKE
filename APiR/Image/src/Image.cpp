#include "Image/Image.h"

Image::Image(
  std::size_t i_width, 
  std::size_t i_height, 
  std::uint32_t /*i_default_color*/)
  : m_width(i_width)
  , m_height(i_height)
  , m_size(i_width * i_height)
  , m_pixels{ MemoryManager::AllocateType::Undefined, nullptr}
  {
  m_pixels = MemoryManager::allocate<uint32_t>(m_size);
  }

Image::~Image()
  {
  MemoryManager::clean(m_pixels);
  }

void Image::SetPixelRGBA(
  std::size_t i_x, 
  std::size_t i_y, 
  uint8_t i_red, 
  uint8_t i_green, 
  uint8_t i_blue, 
  uint8_t i_alpha)
  {
  auto rgba_data = reinterpret_cast<uint8_t*>(&m_pixels.mp_data[_ID(i_x, i_y)]);
  rgba_data[0] = i_red;
  rgba_data[1] = i_green;
  rgba_data[2] = i_blue;
  rgba_data[3] = i_alpha;
  }

uint8_t* Image::GetRGBAData() const
  {
  return reinterpret_cast<uint8_t*>(m_pixels.mp_data);
  }