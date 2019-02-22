#pragma once
#include <stdio.h>
#include <vector>

class BMPWriter
{
public:
  struct Pixel
  {
    Pixel() : red(0), green(0), blue(0) {};
    Pixel(unsigned char i_red, unsigned char i_green, unsigned char i_blue)
      : red(i_red), green(i_green), blue(i_blue)
    {};
    Pixel(const Pixel& i_other)
    {
      red = i_other.red;
      green = i_other.green;
      blue = i_other.blue;
    };
    unsigned char red;
    unsigned char green;
    unsigned char blue;
  };

  struct FileHeader
  {
    unsigned char bfType[2] = { 'B', 'M' };
    unsigned char bfSize[4];
    unsigned char bfReserved1[2] = { 0, 0 };
    unsigned char bfReserved2[2] = { 0, 0 };
    unsigned char bfOffBits[4] = { 54, 0, 0, 0 };
  };

  struct InfoHeader
  {
    unsigned char biSize[4] = { 40, 0, 0, 0 };
    unsigned char biWidth[4];
    unsigned char biHeight[4];
    unsigned char biPlanes[2] = { 1, 0 };
    unsigned char biBitCount[2] = { 24, 0 };
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

private:
  FILE* mp_file;
  size_t m_width;
  size_t m_height;
  size_t m_bytes_per_pixel;
  size_t m_file_size;
  std::vector<std::vector<Pixel>> m_picture;
  FileHeader m_file_header;
  InfoHeader m_info_header;
};