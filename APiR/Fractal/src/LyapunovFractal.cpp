#include "Fractal/LyapunovFractal.h"

#include "Memory/custom_vector.h"

LyapunovFractal::LyapunovFractal(std::string i_fractal_string,
                                 const std::size_t i_width,
                                 const std::size_t i_height,
                                 const std::size_t i_max_iterations)
  : Fractal(i_width, i_height, i_max_iterations)
  , m_fractal_string(std::move(i_fractal_string))
{
  _InitFractalRange();
}

size_t LyapunovFractal::GetValue(const int i_x, const int i_y) const
{
  float zx, zy;
  _MapCoordinate(zx, zy, i_x, i_y);
  return static_cast<size_t>(abs(_ComputeLyapunovExponent(zx, zy) * 255));
  // if (exponent < 0.0)
  //  {
  //  const auto exp = static_cast<uint8_t>(abs(exponent));
  //  return Color(exp, exp, 0);
  //  }
  // else if (exponent == 0.0)
  //  return Color(255, 255, 0);
  // else
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

float LyapunovFractal::_ComputeLyapunovExponent(const float i_zx, const float i_zy) const
{
  custom_vector<float> sequence(m_max_iterations);
  sequence[0] = 0.5f;

  for (std::size_t i = 1; i < m_max_iterations; i++)
    sequence[i] = _MainFunc(i - 1, i_zx, i_zy) * sequence[i - 1] * (1 - sequence[i - 1]);

  for (std::size_t i = 1; i < m_max_iterations; i++)
    sequence[i] = abs((1 - (2 * sequence[i])));

  float result = 0.0f;
  for (std::size_t i = 1; i < m_max_iterations; i++)
    result += logf(abs(_MainFunc(i, i_zx, i_zy)) * sequence[i]);

  return result / static_cast<float>(m_max_iterations);
}

float LyapunovFractal::_MainFunc(const std::size_t i_n, const float i_zx, const float i_zy) const
{
  const std::size_t Sn = i_n % m_fractal_string.size();
  return m_fractal_string[Sn] == 'A' ? i_zx : i_zy;
}
