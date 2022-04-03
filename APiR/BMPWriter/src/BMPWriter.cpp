#include "BMPWriter/BMPWriter.h"
#include "Visual/Color.h"

#include <fstream>

BMPWriter::BMPWriter(ColorMode i_mode)
  : m_color_mode(i_mode)
  {}

void BMPWriter::Write(const std::string& i_file_path, const Image& i_image)
  {
  const auto width  = i_image.GetWidth();
  const auto height = i_image.GetHeight();
  const auto depth  = i_image.GetDepth();

  m_file_header.bfOffBits  = sizeof(FileHeader)  + sizeof(InfoHeader);
  m_file_header.bfSize     = m_file_header.bfOffBits + static_cast<uint32_t>(i_image.GetBytesCount());

  m_info_header.biSize     = sizeof(InfoHeader);
  m_info_header.biHeight   = static_cast<uint32_t>(height);
  m_info_header.biWidth    = static_cast<uint32_t>(width);
  m_info_header.biBitCount = static_cast<uint16_t>(depth * 8);

  std::ofstream file(i_file_path, std::ios::binary);
  file.write(reinterpret_cast<char*>(&m_file_header), sizeof(FileHeader));
  file.write(reinterpret_cast<char*>(&m_info_header), sizeof(InfoHeader));
  for (auto i = 0u; i < height; ++i)
    {
    for (auto j = 0u; j < width; ++j)
      {
      uint8_t r, g, b, a;
      i_image.GetPixelRGBA(j, i, r, g, b, a);
      switch (m_color_mode)
        {
        case BMPWriter::ColorMode::RGBA:
          file << b << g << r << a;
          break;
        case BMPWriter::ColorMode::BGRA:
          file << r << g << b << a;
          break;
        default:
          break;
        }
      }
    }
  }