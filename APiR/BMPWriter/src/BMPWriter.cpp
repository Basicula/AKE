#include <BMPWriter/BMPWriter.h>
#include <Visual/Color.h>

#include <fstream>

BMPWriter::BMPWriter(ColorMode i_mode)
  : m_color_mode(i_mode)
  {}

void BMPWriter::Write(const std::string& i_file_path, const Image& i_image)
  {
  const auto width  = i_image.GetWidth();
  const auto height = i_image.GetHeight();
  const auto depth  = i_image.GetDepth();

  const auto file_header = 14;
  m_file_header.bfOffBits  = file_header  + sizeof(InfoHeader);
  m_file_header.bfSize     = m_file_header.bfOffBits + static_cast<uint32_t>(i_image.GetBytesCount());

  m_info_header.biSize     = sizeof(InfoHeader);
  m_info_header.biHeight   = static_cast<uint32_t>(height);
  m_info_header.biWidth    = static_cast<uint32_t>(width);
  m_info_header.biBitCount = static_cast<uint16_t>(depth * 8);

  std::ofstream file(i_file_path, std::ios::binary);
  file.write(reinterpret_cast<char*>(&m_file_header), 14);
  file.write(reinterpret_cast<char*>(&m_info_header), sizeof(InfoHeader));
  for (auto i = 0u; i < height; ++i)
    {
    for (auto j = 0u; j < width; ++j)
      {
      const auto color_pixel = i_image.GetPixel(j, i);
      const auto r = m_color_mode == ColorMode::BGR ? color_pixel.GetRed() : color_pixel.GetBlue();
      const auto g = color_pixel.GetGreen();
      const auto b = m_color_mode == ColorMode::BGR ? color_pixel.GetBlue() : color_pixel.GetRed();
      const auto a = color_pixel.GetAlpha();
      file << r << g << b << a;
      }
    }
  }