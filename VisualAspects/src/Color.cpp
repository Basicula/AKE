#include "Color.h"

Color::Color()
  : m_red(0)
  , m_green(0)
  , m_blue(0)
{};

Color::Color(unsigned char i_red, unsigned char i_green, unsigned char i_blue)
  : m_red(i_red)
  , m_green(i_green)
  , m_blue(i_blue)
{};

//template<class T>
//Color::Color(T i_red, T i_green, T i_blue)
//  : m_red = std::max(T(0), std::min(i_red, T(255)))
//  , green = std::max(T(0), std::min(i_green, T(255)))
//  , blue = std::max(T(0), std::min(i_blue, T(255)))
//{};

Color::Color(const Color& i_other)
  : m_red(i_other.m_red)
  , m_green(i_other.m_green)
  , m_blue(i_other.m_blue)
{};

bool Color::operator==(const Color& i_other) const
{
  return m_red == i_other.m_red && m_green == i_other.m_green && m_blue == i_other.m_blue;
}

unsigned char* Color::RGB() const
{
  return new unsigned char[3]{ m_red, m_green, m_blue };
};

unsigned char* Color::BGR() const
{
  return new unsigned char[3]{ m_blue, m_green, m_red };
};