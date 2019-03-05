#pragma once
#include <stdio.h>
#include <vector>

#include "BMPUtilities.h"

class BMPWriter
{
public:
  enum ColorMode
  {
    RGB,
    BGR,
    BnW,
  };

public:
  struct FileHeader
  {
    unsigned char bfType[2] = { 'B', 'M' };
    unsigned char bfSize[4];
    unsigned char bfReserved1[2] = { 0, 0 };
    unsigned char bfReserved2[2] = { 0, 0 };
    unsigned char bfOffBits[4] = { 0, 0, 0, 0 };
  };

  struct InfoHeader
  {
    unsigned char biSize[4] = { 0, 0, 0, 0 };
    unsigned char biWidth[4];
    unsigned char biHeight[4];
    unsigned char biPlanes[2] = { 1, 0 };
    unsigned char biBitCount[2] = { 0, 0 };
    unsigned char biCompression[4] = { 0, 0, 0, 0 };
    unsigned char biSizeImage[4] = { 0, 0, 0, 0 };
    unsigned char biXPelsPerMeter[4] = { 0, 0, 0, 0 };
    unsigned char biYPelsPerMeter[4] = { 0, 0, 0, 0 };
    unsigned char biClrUsed[4] = { 0, 0, 0, 0 };
    unsigned char biClrImportant[4] = { 0, 0, 0, 0 };
  };

public:
  BMPWriter(size_t i_width, size_t i_height);
  void Write(const std::string& i_file_path);
  inline void SetPicture(const Picture& i_picture) { m_picture = i_picture; };

private:
  FILE* mp_file;
  size_t m_width;
  size_t m_height;
  size_t m_bytes_per_pixel;
  size_t m_file_size;
  Picture m_picture;
  ColorMode m_color_mode;
  FileHeader m_file_header;
  InfoHeader m_info_header;
};