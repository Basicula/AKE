#include "BMPWriter.h"

BMPWriter::BMPWriter(size_t i_width, size_t i_height)
  : mp_file(nullptr)
  , m_width(i_width)
  , m_height(i_height)
  , m_bytes_per_pixel(3)
  , m_picture(Picture(i_width,i_height))
  , m_color_mode(RGB)
{
  size_t info_size = sizeof(FileHeader) + sizeof(InfoHeader);
  m_file_size = info_size + (m_bytes_per_pixel * i_width + (4 - (m_width*m_bytes_per_pixel) % 4) % 4) * i_height;
  for (auto i = 0u; i < 4; ++i)
  {
    m_file_header.bfSize[i] = static_cast<unsigned char>(m_file_size >> (i * 8));
    m_file_header.bfOffBits[i] = static_cast<unsigned char>(info_size >> (i * 8));
    m_info_header.biHeight[i] = static_cast<unsigned char>(m_height >> (i * 8));
    m_info_header.biWidth[i] = static_cast<unsigned char>(m_width >> (i * 8));
    m_info_header.biSize[i] = static_cast<unsigned char>(sizeof(InfoHeader) >> (i * 8));
  }
  m_info_header.biBitCount[0] = static_cast<unsigned char>(m_bytes_per_pixel * 8);
}

void BMPWriter::Write(const std::string& i_file_path)
{
  unsigned char padding[3] = { 0, 0, 0 };
  int paddingSize = (4 - (m_width*m_bytes_per_pixel) % 4) % 4;
  mp_file = fopen(i_file_path.c_str(), "wb");
  fwrite(&m_file_header, sizeof(FileHeader), 1, mp_file);
  fwrite(&m_info_header, sizeof(InfoHeader), 1, mp_file);
  for (auto i = 0u; i < m_height; ++i)
  {
    if (m_color_mode == BGR)
      fwrite(m_picture[i].data(), sizeof(Pixel), m_picture[i].size(), mp_file);
    else
    {
      for (auto j = 0u; j < m_width; ++j)
      {
        fwrite(m_picture[i][j].BGR(), 1, m_bytes_per_pixel, mp_file);
      }
    }
    fwrite(padding, 1, paddingSize, mp_file);
  }
  fclose(mp_file);
}