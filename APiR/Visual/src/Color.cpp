#include <Color.h>

Color::Color()
  : m_rgb(0)
  {};

Color::Color(std::uint32_t i_rgb)
  : m_rgb(i_rgb)
  {};

Color::Color(std::uint8_t i_red, std::uint8_t i_green, std::uint8_t i_blue)
  : m_rgb(i_red << 16 | i_green << 8 | i_blue)
  {};

Color::Color(const Color& i_other)
  : m_rgb(i_other.m_rgb)
  {};

bool Color::operator==(const Color& i_other) const
  {
  return m_rgb == i_other.m_rgb;
  }

bool Color::operator!=(const Color& i_other) const
  {
  return m_rgb != i_other.m_rgb;
  }

Color Color::operator*(double i_factor) const
  {
  if (i_factor < 0)
    return Color();
  const auto r = static_cast<std::uint8_t>(std::min(255.0, i_factor * GetRed()));
  const auto g = static_cast<std::uint8_t>(std::min(255.0, i_factor * GetGreen()));
  const auto b = static_cast<std::uint8_t>(std::min(255.0, i_factor * GetBlue()));
  return Color(r, g, b);
  }

Color Color::operator*(const Vector3d& i_factor) const
  {
  const auto r = static_cast<std::uint8_t>(std::min(255.0, std::max(0.0, i_factor[0] * GetRed())));
  const auto g = static_cast<std::uint8_t>(std::min(255.0, std::max(0.0, i_factor[1] * GetGreen())));
  const auto b = static_cast<std::uint8_t>(std::min(255.0, std::max(0.0, i_factor[2] * GetBlue())));
  return Color(r, g, b);
  }

Color Color::operator+(const Color& i_other) const
  {
  const auto r = static_cast<std::uint8_t>(std::min(255, int(GetRed()) + int(i_other.GetRed())));
  const auto g = static_cast<std::uint8_t>(std::min(255, int(GetGreen()) + int(i_other.GetGreen())));
  const auto b = static_cast<std::uint8_t>(std::min(255, int(GetBlue()) + int(i_other.GetBlue())));
  return Color(r, g, b);
  }

Color& Color::operator+=(const Color& i_other)
  {
  const auto r = static_cast<std::uint8_t>(std::min(255, int(GetRed()) + int(i_other.GetRed())));
  const auto g = static_cast<std::uint8_t>(std::min(255, int(GetGreen()) + int(i_other.GetGreen())));
  const auto b = static_cast<std::uint8_t>(std::min(255, int(GetBlue()) + int(i_other.GetBlue())));
  m_rgb = (r << 16 | g << 8 | b);
  return *this;
  }
