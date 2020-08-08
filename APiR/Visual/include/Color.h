#pragma once
#include <algorithm>
#include <string>

#include <Vector.h>

class Color
{
public:
  Color();
  Color(std::uint32_t i_rgba);
  Color(
    std::uint8_t i_red, 
    std::uint8_t i_green, 
    std::uint8_t i_blue,
    std::uint8_t i_alpha = 0xff);

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
  std::uint8_t GetAlpha()  const;
  
  void SetRed(std::uint8_t i_red);
  void SetGreen(std::uint8_t i_green);
  void SetBlue(std::uint8_t i_blue);
  void SetAlpha(std::uint8_t i_alpha);
  
  operator std::uint32_t() const;
  std::uint32_t GetRGBA() const;
  void SetRGBA(std::uint32_t i_rgba);

  std::string Serialize() const;
private:
  std::uint32_t m_rgba;
};

inline std::uint8_t Color::GetRed() const
  {
  return reinterpret_cast<const std::uint8_t*>(&m_rgba)[0];
  }
  
inline std::uint8_t Color::GetGreen() const
  { 
  return reinterpret_cast<const std::uint8_t*>(&m_rgba)[1];
  }
  
inline std::uint8_t Color::GetBlue() const
  { 
  return reinterpret_cast<const std::uint8_t*>(&m_rgba)[2];
  }
inline std::uint8_t Color::GetAlpha() const
  {
  return reinterpret_cast<const std::uint8_t*>(&m_rgba)[3];
  }
  
inline void Color::SetRed(std::uint8_t i_red)
  {
  reinterpret_cast<std::uint8_t*>(&m_rgba)[0] = i_red;
  }

inline void Color::SetGreen(std::uint8_t i_green)
  {
  reinterpret_cast<std::uint8_t*>(&m_rgba)[1] = i_green;
  }
  
inline void Color::SetBlue(std::uint8_t i_blue)
  {
  reinterpret_cast<std::uint8_t*>(&m_rgba)[2] = i_blue;
  }

inline void Color::SetAlpha(std::uint8_t i_alpha)
  {
  reinterpret_cast<std::uint8_t*>(&m_rgba)[3] = i_alpha;
  }
  
inline Color::operator std::uint32_t() const
  {
  return m_rgba;
  }

inline std::uint32_t Color::GetRGBA() const
  {
  return m_rgba;
  }
  
inline void Color::SetRGBA(std::uint32_t i_rgba)
  {
  m_rgba = i_rgba;
  }

inline std::string Color::Serialize() const
  {
  return "{ \"Color\" : " + std::to_string(m_rgba) + " }";
  }