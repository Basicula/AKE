#pragma once
#include <algorithm>
#include <string>

#include <Vector.h>

class Color
{
public:
  Color();
  Color(std::uint32_t i_rgb);
  Color(std::uint8_t i_red, std::uint8_t i_green, std::uint8_t i_blue);

  Color(const Color& i_other);

  Color operator*(double i_factor) const;
  Color operator*(const Vector3d& i_factor) const;
  Color operator+(const Color& i_other) const;
  Color& operator+=(const Color& i_other);

  bool operator==(const Color& i_other) const;
  bool operator!=(const Color& i_other) const;
  
  std::uint8_t GetRed()   const;
  std::uint8_t GetGreen() const;
  std::uint8_t GetBlue()  const;
  
  void SetRed(std::uint8_t i_red);
  void SetGreen(std::uint8_t i_green);
  void SetBlue(std::uint8_t i_blue);
  
  operator std::uint32_t();
  std::uint32_t GetRGB() const;
  void SetRGB(std::uint32_t i_rgb);

  std::string Serialize() const;
private:
  std::uint32_t m_rgb;
};

inline std::uint8_t Color::GetRed() const
  {
  return static_cast<std::uint8_t>((m_rgb & 0xff0000) >> 16);
  }
  
inline std::uint8_t Color::GetGreen() const
  { 
  return static_cast<std::uint8_t>((m_rgb & 0x00ff00) >> 8 );
  };
  
inline std::uint8_t Color::GetBlue() const
  { 
  return static_cast<std::uint8_t>(m_rgb & 0x0000ff);
  };
  
inline void Color::SetRed(std::uint8_t i_red)
  {
  m_rgb &= 0x00ffff;
  m_rgb |= (i_red << 16);
  }

inline void Color::SetGreen(std::uint8_t i_green)
  {
  m_rgb &= 0xff00ff;
  m_rgb |= (i_green << 8);
  }
  
inline void Color::SetBlue(std::uint8_t i_blue)
  {
  m_rgb &= 0xffff00;
  m_rgb |= i_blue;
  }
  
inline Color::operator std::uint32_t()
  {
  return m_rgb;
  }

inline std::uint32_t Color::GetRGB() const
  {
  return m_rgb;
  }
  
inline void Color::SetRGB(std::uint32_t i_rgb)
  {
  m_rgb = i_rgb;
  }

inline std::string Color::Serialize() const
  {
  return "{ \"Color\" : " + std::to_string(m_rgb) + " }";
  }