#include <Visual/Color.h>

namespace
  {
  bool IsBigEndian()
    {
    int a = 1;
    return (reinterpret_cast<char*>(&a)[0] == 0);
    }
  }

Color::Color()
  : m_rgba(0)
  {};

Color::Color(std::uint32_t i_rgba)
  : m_rgba(i_rgba)
  {
  if (!IsBigEndian())
    {
    auto rgba_data = reinterpret_cast<std::uint8_t*>(&m_rgba);
    std::reverse(rgba_data, rgba_data + sizeof(m_rgba));
    }
  };

Color::Color(
  std::uint8_t i_red, 
  std::uint8_t i_green, 
  std::uint8_t i_blue,
  std::uint8_t i_alpha)
  : Color()
  {
  auto rgba_data = reinterpret_cast<std::uint8_t*>(&m_rgba);
  rgba_data[0] = i_red;
  rgba_data[1] = i_green;
  rgba_data[2] = i_blue;
  rgba_data[3] = i_alpha;
  };

Color::Color(const Color& i_other)
  : m_rgba(i_other.m_rgba)
  {};

bool Color::operator==(const Color& i_other) const
  {
  return m_rgba == i_other.m_rgba;
  }

bool Color::operator!=(const Color& i_other) const
  {
  return m_rgba != i_other.m_rgba;
  }

Color Color::operator*(double i_factor) const
  {
  if (i_factor < 0)
    return Color();
  const auto r = static_cast<std::uint8_t>(std::min(255.0, i_factor * GetRed()));
  const auto g = static_cast<std::uint8_t>(std::min(255.0, i_factor * GetGreen()));
  const auto b = static_cast<std::uint8_t>(std::min(255.0, i_factor * GetBlue()));
  return Color(r, g, b, GetAlpha());
  }

Color Color::operator*(const Vector3d& i_factor) const
  {
  const auto r = static_cast<std::uint8_t>(std::min(255.0, std::max(0.0, i_factor[0] * GetRed())));
  const auto g = static_cast<std::uint8_t>(std::min(255.0, std::max(0.0, i_factor[1] * GetGreen())));
  const auto b = static_cast<std::uint8_t>(std::min(255.0, std::max(0.0, i_factor[2] * GetBlue())));
  return Color(r, g, b, GetAlpha());
  }

Color Color::operator+(const Color& i_other) const
  {
  const auto r = static_cast<std::uint8_t>(std::min(255, int(GetRed()) + int(i_other.GetRed())));
  const auto g = static_cast<std::uint8_t>(std::min(255, int(GetGreen()) + int(i_other.GetGreen())));
  const auto b = static_cast<std::uint8_t>(std::min(255, int(GetBlue()) + int(i_other.GetBlue())));
  const auto a = static_cast<std::uint8_t>(std::min(255, int(GetAlpha()) + int(i_other.GetAlpha())));
  return Color(r, g, b);
  }

Color& Color::operator+=(const Color& i_other)
  {
  auto rgba_data = reinterpret_cast<std::uint8_t*>(&m_rgba);
  rgba_data[0] = static_cast<std::uint8_t>(std::min(255, int(GetRed()) + int(i_other.GetRed())));
  rgba_data[1] = static_cast<std::uint8_t>(std::min(255, int(GetGreen()) + int(i_other.GetGreen())));
  rgba_data[2] = static_cast<std::uint8_t>(std::min(255, int(GetBlue()) + int(i_other.GetBlue())));
  rgba_data[3] = static_cast<std::uint8_t>(std::min(255, int(GetAlpha()) + int(i_other.GetAlpha())));
  return *this;
  }
