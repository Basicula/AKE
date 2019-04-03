#pragma once

#include <vector>

namespace Equations
  {
  /*
  Equations representation looks like:
  a0 + a1*x + ... + an*x^n
  So coefs : a0 <=> coefs[0] and so on

  if functions returns empty vector, the equation can't be solved in rational numbers or given coefs are given in wrong way
  */
  //i_coefs must have 3 elements
  int SolveQuadratic(const double* i_coefs, double *io_roots);
  //i_coefs must have 4 elements
  int SolveCubic(const double* i_coefs, double *io_roots);
  //i_coefs must have 5 elements
  int SolveQuartic(const double* i_coefs, double *io_roots);
  }
