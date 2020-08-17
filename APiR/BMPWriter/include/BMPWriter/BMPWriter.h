#pragma once
#include <Rendering/Image.h>

class BMPWriter
{
public:
  enum class ColorMode
  {
    RGBA,
    BGRA,
  };

public:
#pragma pack(1)
  struct FileHeader
  {
    uint16_t bfType = 0x4d42; // BM
    uint32_t bfSize = 0;      // files size in bytes
    uint16_t bfReserved1 = 0; // reserved 2 bytes
    uint16_t bfReserved2 = 0; // reserved 2 bytes
    uint32_t bfOffBits = 0;   // offset to pixel data
  };
#pragma pack()

#pragma pack(1)
  struct InfoHeader
  {
    uint32_t biSize;              // sizeof current struct
    uint32_t biWidth;             // width of image
    uint32_t biHeight;            // height of image
    uint16_t biPlanes = 1;        // just 1 :)
    uint16_t biBitCount;          // bits per color
    uint32_t biCompression = 0;   // compression method
    uint32_t biSizeImage = 0;     // sizeof pixels in bytes but we skip it
    uint32_t biXPelsPerMeter = 0; // unused vars
    uint32_t biYPelsPerMeter = 0; // unused vars
    uint32_t biClrUsed = 0;       // unused vars
    uint32_t biClrImportant = 0;  // unused vars
  };
#pragma pack()

public:
  BMPWriter(ColorMode i_mode = ColorMode::RGBA);
  void Write(const std::string& i_file_path, const Image& i_image);

private:
  ColorMode m_color_mode;
  FileHeader m_file_header;
  InfoHeader m_info_header;
};