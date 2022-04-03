#include "Math/SolveEquations.h"

#include "Math/Constants.h"

#include <cmath>

namespace Equations {
  double CubicRoot(const double i_val)
  {
    return i_val > 0.0 ? std::pow(i_val, 1.0 / 3.0) : (i_val < 0.0 ? -std::pow(-i_val, 1.0 / 3.0) : 0.0);
  }

  int SolveQuadratic(const double* i_coefs, double* io_roots)
  {
    if (i_coefs[1] == 0.0 && i_coefs[2] == 0.0)
      return 0;

    const double common_solution = -i_coefs[1] / i_coefs[2];
    if (i_coefs[0] == 0.0) {
      io_roots[0] = common_solution;
      io_roots[1] = 0.0;
      return 2;
    }

    const double square_descriminant = i_coefs[1] * i_coefs[1] - 4.0 * i_coefs[0] * i_coefs[2];
    if (square_descriminant < 0.0)
      return 0;

    if (square_descriminant == 0.0) {
      io_roots[0] = common_solution / 2.0;
      return 1;
    }

    const double descriminant = std::sqrt(square_descriminant);
    io_roots[0] = (descriminant - i_coefs[1]) / (2.0 * i_coefs[2]);
    io_roots[1] = -(descriminant + i_coefs[1]) / (2.0 * i_coefs[2]);
    return 2;
  }

  int SolveCubic(const double* i_coefs, double* io_roots)
  {
    if (i_coefs[0] == 0.0) {
      int roots = SolveQuadratic(i_coefs + 1, io_roots);
      io_roots[roots++] = 0.0;
      return roots;
    }

    int roots_count;
    // transform to x^3 + A*x^2 + B*x + C = 0
    const double A = i_coefs[2] / i_coefs[3];
    const double B = i_coefs[1] / i_coefs[3];
    const double C = i_coefs[0] / i_coefs[3];

    // transform to x^3 + q*x + p = 0 by replace x = y - A/3
    const double p = B - A * A / 3.0;
    const double q = C + 2.0 * A * A * A / 27.0 - A * B / 3.0;

    /*
    now let x = u + v =>
            x^3 = (u + v)^3 =>
            x^3 = u^3 + v^3 + 3*u*v*(u+v) =>
            x^3 - 3*u*v*x - (u^3 + v^3) = 0 =>
            p = -3*u*v , q = -(u^3 + v^3) =>
            u^3 * v^3 = -p^3/27
    we have simple representation for u^3 and v^3 like roots for quadratic equation as x^2 - (x1+x2)*x + x1*x2 = 0 where
    x1,x2 are roots of this equation so from x^2 + qx - p^3/27 = 0 have D = q^2 + 4*p^3/27 or D = 27*q^2 + 4*p^3 or D =
    q^2 / 4 + p^3/27
    */
    const double D = q * q / 4.0 + p * p * p / 27.0;

    /*
    Special case when D=0 we have one root 3 * q / p and double root -3 * q / 2 * p
    so if q = 0 we have triple root 0
    */
    if (D == 0.0) {
      if (q == 0.0) {
        io_roots[0] = 0.0;
        roots_count = 1;
      } else {
        io_roots[0] = 3.0 * q / p;
        io_roots[1] = -3.0 * q / (2.0 * p);
        roots_count = 2;
      }
    } else if (D > 0) {
      io_roots[0] = CubicRoot(-q / 2.0 + std::sqrt(D)) + CubicRoot(-q / 2.0 - std::sqrt(D));
      roots_count = 1;
    } else {
      const double angle = std::acos(3.0 * q * std::sqrt(-3.0 / p) / (2.0 * p)) / 3.0;
      const double t = 2.0 * std::sqrt(-p / 3.0);
      io_roots[0] = t * std::cos(angle);
      io_roots[1] = t * std::cos(angle - 2.0 * Math::Constants::PI / 3.0);
      io_roots[2] = t * std::cos(angle + 2.0 * Math::Constants::PI / 3.0);
      roots_count = 3;
    }

    // transform from x^3 + px + q = 0 to x^3 + Ax^2 + Bx + C = 0
    const double sub = A / 3.0;
    for (auto i = 0; i < roots_count; ++i)
      io_roots[i] -= sub;

    return roots_count;
  }

  int SolveQuartic(const double* i_coefs, double* io_roots)
  {
    // transform to x^4 + Ax^3 + Bx^2 + Cx + D = 0
    const double A = i_coefs[3] / i_coefs[4];
    const double B = i_coefs[2] / i_coefs[4];
    const double C = i_coefs[1] / i_coefs[4];
    const double D = i_coefs[0] / i_coefs[4];

    // transform to x^4 + px^2 + qx + r = 0 by replace x = y - A/4
    const double square_A = A * A;
    const double p = B - 3.0 * square_A / 8.0;
    const double q = C - A * B / 2.0 + square_A * A / 8.0;
    const double r = D - A * C / 4.0 + square_A * B / 16.0 - 3.0 * square_A * square_A / 256.0;

    int roots_count;
    if (r == 0.0) {
      /*
      special case when r = 0
      x^4 + px^2 + qx + r = 0 =>
      x(x^3 + px + q) = 0
      */
      const double cubic_coefs[4] = { q, p, 0.0, 1.0 };
      roots_count = SolveCubic(cubic_coefs, io_roots);
      io_roots[roots_count++] = 0.0;
    } else {
      /*
      x^4 + px^2 + qx + r = 0 =>
      x^4 + px^2 = -qx - r =>
      (x^2)^2 + 2 * (p / 2) * x^2 + (p/2)^2 = -qx - r =>
      (x^2 + p/2)^2 = -qx - r + p^2 / 4 =>
      (x^2 + p/2 + m)^2 = m^2 + 2*m*x^2 + mp -qx - r + p^2/4 =>
      try make perfect square on the right side =>
      (x^2 + p/2 + m)^2 = 2*m*x^2 -qx + m^2 + mp +p^2/4 - r =>
      D = q^2 - 4 * (2*m) * (m^2 + mp +p^2/4 - r) must be equal 0 =>
      -8*m^3 - 8*p*m^2 - 8*m*(p^2/4 - r) + q^2 = 0 =>
      8*m^3 + 8*p*m^2 + m*(2*p^2 - 8*r) - q^2 = 0

      then we have (x^2 + p/2 + m)^2 = (sqrt(2*m)*x - q / (2 * sqrt(2*m))^2 => m > 0 =>
      (x^2 + sqrt(2*m)*x + p/2 + m - q / (2 * sqrt(2*m))*(x^2 - sqrt(2*m)*x + p/2 + m + q / (2 * sqrt(2*m)) = 0
      */
      const double cubic_coefs[4] = { -q * q, 2.0 * p * p - 8.0 * r, 8.0 * p, 8.0 };
      double cubic_roots[3];
      const int cubic_roots_count = SolveCubic(cubic_coefs, cubic_roots);
      double cubic_res = cubic_roots[0];
      for (auto i = 1; cubic_res <= 0.0 && i < cubic_roots_count; ++i)
        cubic_res = cubic_roots[i];
      const double sqrt_double_cubic_res = std::sqrt(2.0 * cubic_res);
      const double quadratic_coefs1[3] = { p / 2.0 + cubic_res - q / (2.0 * sqrt_double_cubic_res),
                                           sqrt_double_cubic_res,
                                           1.0 };
      roots_count = SolveQuadratic(quadratic_coefs1, io_roots);
      const double quadratic_coefs2[3] = { p / 2.0 + cubic_res + q / (2.0 * sqrt_double_cubic_res),
                                           -sqrt_double_cubic_res,
                                           1.0 };
      roots_count += SolveQuadratic(quadratic_coefs2, io_roots + roots_count);
    }

    const double sub = A / 4.0;
    for (auto i = 0; i < roots_count; ++i)
      io_roots[i] -= sub;
    return roots_count;
  }
}