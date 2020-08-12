#pragma once
#include <Visual/Image.h>

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
    uint16_t bfType = 0x4d42; // BM
    uint32_t bfSize = 0;          // files size in bytes
    uint32_t bfReserved = 0;  // reserved 4 bytes Reserved1 and Reserved2
    uint32_t bfOffBits = 0;       // offset to pixel data
  };

  struct InfoHeader
  {
    uint32_t biSize;              // sizeof current struct
    uint32_t biWidth;             // width of image
    uint32_t biHeight;            // height of image
    uint16_t biPlanes = 1;        // just 1 :)
    uint16_t biBitCount;          // bits per color
    uint32_t biCompression = 6;   // compression method
    uint32_t biSizeImage = 0;     // sizeof pixels in bytes but we skip it
    uint32_t biXPelsPerMeter = 0; // unused vars
    uint32_t biYPelsPerMeter = 0; // unused vars
    uint32_t biClrUsed = 0;       // unused vars
    uint32_t biClrImportant = 0;  // unused vars
  };

public:
  BMPWriter(ColorMode i_mode = ColorMode::RGB);
  void Write(const std::string& i_file_path, const Image& i_image);

private:
  ColorMode m_color_mode;
  FileHeader m_file_header;
  InfoHeader m_info_header;
};