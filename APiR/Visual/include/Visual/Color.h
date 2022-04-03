#pragma once
#include "Macros.h"
#include "Math/Vector.h"

class Color
{
public:
  HOSTDEVICE Color();
  HOSTDEVICE Color(std::uint32_t i_abgr);
  HOSTDEVICE Color(uint8_t i_red, uint8_t i_green, uint8_t i_blue, uint8_t i_alpha = 0xff);

  HOSTDEVICE Color(const Color& i_other);
  HOSTDEVICE Color& operator=(const Color& i_other);

  HOSTDEVICE Color operator*(double i_factor) const;
  HOSTDEVICE Color operator*(const Vector3d& i_factor) const;
  HOSTDEVICE Color operator+(const Color& i_other) const;
  HOSTDEVICE Color& operator+=(const Color& i_other);

  bool operator==(const Color& i_other) const;
  bool operator!=(const Color& i_other) const;

  HOSTDEVICE uint8_t GetRed() const;
  HOSTDEVICE uint8_t GetGreen() const;
  HOSTDEVICE uint8_t GetBlue() const;
  HOSTDEVICE uint8_t GetAlpha() const;

  void SetRed(uint8_t i_red);
  void SetGreen(uint8_t i_green);
  void SetBlue(uint8_t i_blue);
  void SetAlpha(uint8_t i_alpha);

  HOSTDEVICE operator std::uint32_t() const;
  std::uint32_t GetRGBA() const;
  void SetRGBA(std::uint32_t i_rgba);

  std::string Serialize() const;

public:
  static Color RandomColor();

  static const Color White;
  static const Color Black;
  static const Color Red;
  static const Color Green;
  static const Color Blue;
  static const Color Yellow;

private:
  std::uint32_t m_rgba;
};

inline uint8_t Color::GetRed() const
{
  return reinterpret_cast<const uint8_t*>(&m_rgba)[0];
}

inline uint8_t Color::GetGreen() const
{
  return reinterpret_cast<const uint8_t*>(&m_rgba)[1];
}

inline uint8_t Color::GetBlue() const
{
  return reinterpret_cast<const uint8_t*>(&m_rgba)[2];
}
inline uint8_t Color::GetAlpha() const
{
  return reinterpret_cast<const uint8_t*>(&m_rgba)[3];
}

inline void Color::SetRed(uint8_t i_red)
{
  reinterpret_cast<uint8_t*>(&m_rgba)[0] = i_red;
}

inline void Color::SetGreen(uint8_t i_green)
{
  reinterpret_cast<uint8_t*>(&m_rgba)[1] = i_green;
}

inline void Color::SetBlue(uint8_t i_blue)
{
  reinterpret_cast<uint8_t*>(&m_rgba)[2] = i_blue;
}

inline void Color::SetAlpha(uint8_t i_alpha)
{
  reinterpret_cast<uint8_t*>(&m_rgba)[3] = i_alpha;
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