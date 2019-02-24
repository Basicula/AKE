#include "BMPUtilities.h"

void DrawLine(Picture& io_picture, int x0, int y0, int x1, int y1)
{
  bool inverse = false;
  if (abs(x1 - x0) < abs(y1 - y0))
  {
    std::swap(x0, y0);
    std::swap(x1, y1);
    inverse = true;
  }

  if (x1 < x0)
  {
    std::swap(x0, x1);
    std::swap(y0, y1);
  }

  int dx = x1 - x0;
  int dy = y1 - y0;
  int derr = 2 * abs(dy);
  int err = 0;

  for (int x = x0, y = y0; x <= x1; ++x)
  {
    if (inverse)
      io_picture[x][y] = Pixel(255, 255, 255);
    else
      io_picture[y][x] = Pixel(255, 255, 255);

    err += derr;

    if (err > dx)
    {
      y += (y1 > y0 ? 1 : -1);
      err -= dx * 2;
    }
  }
}

Picture DiagonalLine(size_t i_width, size_t i_height)
{
  Picture res = Picture(i_width, i_height);
  DrawLine(res, 0, 0, i_width - 1, i_height - 1);
  return res;
}

Picture MandelbrotSet(size_t i_width, size_t i_height)
{
  Picture res = Picture(i_width, i_height);
  const int max_iterations = 1000;
  for (size_t y = 0; y < i_height; ++y)
    for (size_t x = 0; x < i_width; ++x)
    {
      double cx = 3.5 * x / i_width - 2.5;
      double cy = 2.0 * y / i_height - 1.0;
      double zx = 0;
      double zy = 0;
      int iter = 0;
      while (zx * zx + zy * zy <= 4 && iter < max_iterations)
      {
        double tempzx = zx * zx - zy * zy + cx;
        zy = 2 * zx * zy + cy;
        zx = tempzx;
        ++iter;
      }
      res[y][x] = Pixel(sin(iter) * sin(iter)*255, cos(iter) * cos(iter)*255, 0);
    }
  return res;
}