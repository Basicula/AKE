#include <fstream>

#include <BMPWriter.h>
#include <Color.h>

BMPWriter::BMPWriter(ColorMode i_mode)
  : m_color_mode(i_mode)
  {}

void BMPWriter::Write(const std::string& i_file_path, const Image& i_image)
  {
  const auto width = i_image.GetWidth();
  const auto height = i_image.GetHeight();
  const auto depth = i_image.GetDepth();

  const auto info_size = sizeof(FileHeader) + sizeof(InfoHeader);
  const auto file_size = info_size + depth * width * height;
  for (auto i = 0u; i < 4; ++i)
    {
    m_file_header.bfSize[i] = static_cast<unsigned char>(file_size >> (i * 8));
    m_file_header.bfOffBits[i] = static_cast<unsigned char>(info_size >> (i * 8));
    m_info_header.biHeight[i] = static_cast<unsigned char>(height >> (i * 8));
    m_info_header.biWidth[i] = static_cast<unsigned char>(width >> (i * 8));
    m_info_header.biSize[i] = static_cast<unsigned char>(sizeof(InfoHeader) >> (i * 8));
    }
  m_info_header.biBitCount[0] = static_cast<unsigned char>(depth * 8);
  unsigned char padding[3] = { 0, 0, 0 };
  int paddingSize = (4 - (width * depth) % 4) % 4;
  fopen_s(&mp_file, i_file_path.c_str(), "wb");
  fwrite(&m_file_header, sizeof(FileHeader), 1, mp_file);
  fwrite(&m_info_header, sizeof(InfoHeader), 1, mp_file);
  for (auto i = 0u; i < height; ++i)
    {
    for (auto j = 0u; j < width; ++j)
      {
      const auto color_pixel = Color(i_image.GetPixel(j, i));
      const auto r = m_color_mode == ColorMode::BGR ? color_pixel.GetRed() : color_pixel.GetBlue();
      const auto g = color_pixel.GetGreen();
      const auto b = m_color_mode == ColorMode::BGR ? color_pixel.GetBlue() : color_pixel.GetRed();
      fwrite(&r, 1, 1, mp_file);
      fwrite(&g, 1, 1, mp_file);
      fwrite(&b, 1, 1, mp_file);
      }
    fwrite(padding, 1, paddingSize, mp_file);
    }
  fclose(mp_file);
  }