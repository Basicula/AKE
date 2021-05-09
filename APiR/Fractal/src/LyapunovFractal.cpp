#include <Fractal/LyapunovFractal.h>

LyapunovFractal::LyapunovFractal(
  const std::string& i_fractal_string,
  std::size_t i_width,
  std::size_t i_height,
  std::size_t i_max_iterations)
  : Fractal(i_width, i_height, i_max_iterations)
  , m_fractal_string(i_fractal_string)
  {
  _InitFractalRange();
  }

size_t LyapunovFractal::GetValue(int i_x, int i_y) const
  {
  double zx, zy;
  _MapCoordinate(zx, zy, i_x, i_y);
  return static_cast<size_t>(abs(_ComputeLyapunovExponent(zx, zy) * 255));
  //if (exponent < 0.0)
  //  {
  //  const auto exp = static_cast<uint8_t>(abs(exponent));
  //  return Color(exp, exp, 0);
  //  }
  //else if (exponent == 0.0)
  //  return Color(255, 255, 0);
  //else
  //  {
  //  auto exp = static_cast<uint8_t>(exponent);
  //  return Color(0, exp / 2, exp);
  //  }
  }

void LyapunovFractal::_InitFractalRange()
  {
  m_x_min = 0.0;
  m_x_max = 4.0;
  m_y_min = 0.0;
  m_y_max = 4.0;
  }

double LyapunovFractal::_ComputeLyapunovExponent(double i_zx, double i_zy) const
  {
  custom_vector<double> sequence(m_max_iterations);
  sequence[0] = 0.5;

  for (std::size_t i = 1; i < m_max_iterations; i++)
    sequence[i] = _MainFunc(i - 1, i_zx, i_zy) * sequence[i - 1] * (1 - sequence[i - 1]);

  for (std::size_t i = 1; i < m_max_iterations; i++) 
    sequence[i] = abs((1 - (2 * sequence[i])));

  double result = 0.0;
  for (std::size_t i = 1; i < m_max_iterations; i++)
    result += log(abs(_MainFunc(i, i_zx, i_zy)) * sequence[i]);

  return result / m_max_iterations;
  }

double LyapunovFractal::_MainFunc(std::size_t i_n, double i_zx, double i_zy) const
  {
  const std::size_t Sn = i_n % m_fractal_string.size();
  return m_fractal_string[Sn] == 'A' ? i_zx : i_zy;
  }
