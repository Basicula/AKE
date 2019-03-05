#include <omp.h>

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

Picture MandelbrotSet(size_t i_width, size_t i_height, RunMode i_run_mode)
{
  Picture res = Picture(i_width, i_height);
  const int max_iterations = 10000;
  int begin,end;
  if (i_run_mode == MPI)
  {
    /*int size, rank;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("rank,size %d %d",size,rank);
    int n = (i_height - 1)/size + 1;
    begin = rank*n;
    end = (rank+1)*n;*/
  }
  else
  {
    begin = 0;
    end = i_height;
  }
  int omp_val = 0;
  if (i_run_mode == OMP)
  {
    omp_val = 8;
  }
#pragma omp parallel for if(omp_val) default(none) shared(res) num_threads(8)
  for (int y = begin; y < end; ++y)
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
  if (i_run_mode == MPI)
  {
    /*MPI_Finalize();*/
  }
  return res;
}