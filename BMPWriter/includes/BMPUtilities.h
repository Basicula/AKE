#pragma once
#include <vector>

struct Pixel
{
  Pixel() : red(0), green(0), blue(0) {};
  Pixel(unsigned char i_red, unsigned char i_green, unsigned char i_blue) : red(i_red), green(i_green), blue(i_blue) {};
  Pixel(const Pixel& i_other)
  {
    red = i_other.red;
    green = i_other.green;
    blue = i_other.blue;
  };
  inline unsigned char* RGB() const { return new unsigned char[3]{ red, green, blue }; };
  inline unsigned char* BGR() const { return new unsigned char[3]{ blue, green, red }; };
  unsigned char red;
  unsigned char green;
  unsigned char blue;
};

using Picture = std::vector<std::vector<Pixel>>;
#define Picture(i_width,i_height) std::vector<std::vector<Pixel>>(i_height, std::vector<Pixel>(i_width))

Picture DiagonalLine(size_t i_width, size_t i_height);
Picture MandelbrotSet(size_t i_width, size_t i_height);