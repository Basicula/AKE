#pragma once
#include <Visual/Image.h>

#include <stdio.h>
#include <vector>

class BMPWriter
{
public:
  enum class ColorMode
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
  BMPWriter(ColorMode i_mode = ColorMode::RGB);
  void Write(const std::string& i_file_path, const Image& i_image);

private:
  FILE* mp_file;
  ColorMode m_color_mode;
  FileHeader m_file_header;
  InfoHeader m_info_header;
};