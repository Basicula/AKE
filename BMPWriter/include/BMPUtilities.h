#pragma once
#include <vector>

#include <Color.h>

enum RunMode
{
  STANDART,
  OMP,
  MPI,
};

using Picture = std::vector<std::vector<Color>>;
#define Picture(i_width,i_height) std::vector<std::vector<Color>>(i_height, std::vector<Color>(i_width))

Picture DiagonalLine(size_t i_width, size_t i_height);
Picture MandelbrotSet(size_t i_width, size_t i_height, RunMode i_run_mode = STANDART);