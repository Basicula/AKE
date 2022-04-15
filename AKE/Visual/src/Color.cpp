#include "Visual/Color.h"

#include "Common/Randomizer.h"
#include "Common/Utils.h"

const Color Color::White = Color(255, 255, 255);
const Color Color::Black = Color(0, 0, 0);
const Color Color::Blue = Color(0, 0, 255);
const Color Color::Green = Color(0, 255, 0);
const Color Color::Red = Color(255, 0, 0);
const Color Color::Yellow = Color(255, 255, 0);

Color::Color()
  : m_rgba(0){};

Color::Color(const std::uint32_t i_abgr)
  : m_rgba(i_abgr){};

Color::Color(const std::uint8_t i_red,
             const std::uint8_t i_green,
             const std::uint8_t i_blue,
             const std::uint8_t i_alpha)
  : Color()
{
  auto* rgba_data = reinterpret_cast<std::uint8_t*>(&m_rgba);
  rgba_data[0] = i_red;
  rgba_data[1] = i_green;
  rgba_data[2] = i_blue;
  rgba_data[3] = i_alpha;
};

Color& Color::operator=(const Color& i_other)
{
  if (this == &i_other) {
    return *this;
  }
  m_rgba = i_other.m_rgba;
  return *this;
};

bool Color::operator==(const Color& i_other) const
{
  return m_rgba == i_other.m_rgba;
}

bool Color::operator!=(const Color& i_other) const
{
  return m_rgba != i_other.m_rgba;
}

Color Color::operator*(const double i_factor) const
{
  if (i_factor < 0)
    return {};
  const auto r = static_cast<std::uint8_t>(Utils::min(255.0, i_factor * GetRed()));
  const auto g = static_cast<std::uint8_t>(Utils::min(255.0, i_factor * GetGreen()));
  const auto b = static_cast<std::uint8_t>(Utils::min(255.0, i_factor * GetBlue()));
  return { r, g, b, GetAlpha() };
}

Color Color::operator*(const Vector3d& i_factor) const
{
  const auto r = static_cast<std::uint8_t>(Utils::min(255.0, Utils::max(0.0, i_factor[0] * GetRed())));
  const auto g = static_cast<std::uint8_t>(Utils::min(255.0, Utils::max(0.0, i_factor[1] * GetGreen())));
  const auto b = static_cast<std::uint8_t>(Utils::min(255.0, Utils::max(0.0, i_factor[2] * GetBlue())));
  return { r, g, b, GetAlpha() };
}

Color Color::operator+(const Color& i_other) const
{
  const auto r =
    static_cast<std::uint8_t>(Utils::min(255, static_cast<int>(GetRed()) + static_cast<int>(i_other.GetRed())));
  const auto g =
    static_cast<std::uint8_t>(Utils::min(255, static_cast<int>(GetGreen()) + static_cast<int>(i_other.GetGreen())));
  const auto b =
    static_cast<std::uint8_t>(Utils::min(255, static_cast<int>(GetBlue()) + static_cast<int>(i_other.GetBlue())));
  const auto a =
    static_cast<std::uint8_t>(Utils::min(255, static_cast<int>(GetAlpha()) + static_cast<int>(i_other.GetAlpha())));
  return { r, g, b };
}

Color& Color::operator+=(const Color& i_other)
{
  auto* rgba_data = reinterpret_cast<std::uint8_t*>(&m_rgba);
  rgba_data[0] =
    static_cast<std::uint8_t>(Utils::min(255, static_cast<int>(GetRed()) + static_cast<int>(i_other.GetRed())));
  rgba_data[1] =
    static_cast<std::uint8_t>(Utils::min(255, static_cast<int>(GetGreen()) + static_cast<int>(i_other.GetGreen())));
  rgba_data[2] =
    static_cast<std::uint8_t>(Utils::min(255, static_cast<int>(GetBlue()) + static_cast<int>(i_other.GetBlue())));
  rgba_data[3] =
    static_cast<std::uint8_t>(Utils::min(255, static_cast<int>(GetAlpha()) + static_cast<int>(i_other.GetAlpha())));
  return *this;
}

Color Color::RandomColor()
{
  static Randomizer color_randomizer;
  return { color_randomizer.Next<std::uint8_t>(),
           color_randomizer.Next<std::uint8_t>(),
           color_randomizer.Next<std::uint8_t>() };
}

std::uint8_t Color::GetRed() const
{
  return reinterpret_cast<const std::uint8_t*>(&m_rgba)[0];
}

std::uint8_t Color::GetGreen() const
{
  return reinterpret_cast<const std::uint8_t*>(&m_rgba)[1];
}

std::uint8_t Color::GetBlue() const
{
  return reinterpret_cast<const std::uint8_t*>(&m_rgba)[2];
}
std::uint8_t Color::GetAlpha() const
{
  return reinterpret_cast<const std::uint8_t*>(&m_rgba)[3];
}

void Color::SetRed(const std::uint8_t i_red)
{
  reinterpret_cast<std::uint8_t*>(&m_rgba)[0] = i_red;
}

void Color::SetGreen(const std::uint8_t i_green)
{
  reinterpret_cast<std::uint8_t*>(&m_rgba)[1] = i_green;
}

void Color::SetBlue(const std::uint8_t i_blue)
{
  reinterpret_cast<std::uint8_t*>(&m_rgba)[2] = i_blue;
}

void Color::SetAlpha(const std::uint8_t i_alpha)
{
  reinterpret_cast<std::uint8_t*>(&m_rgba)[3] = i_alpha;
}

Color::operator std::uint32_t() const
{
  return m_rgba;
}

std::uint32_t Color::GetRGBA() const
{
  return m_rgba;
}

void Color::SetRGBA(const std::uint32_t i_rgba)
{
  m_rgba = i_rgba;
}

std::string Color::Serialize() const
{
  return "{ \"Color\" : " + std::to_string(m_rgba) + " }";
}