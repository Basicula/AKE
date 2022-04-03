#pragma once
#include "Macros.h"
#include "Math/Vector.h"

class Color
{
public:
  HOSTDEVICE Color();
  HOSTDEVICE explicit Color(std::uint32_t i_abgr);
  HOSTDEVICE Color(std::uint8_t i_red, std::uint8_t i_green, std::uint8_t i_blue, std::uint8_t i_alpha = 0xff);

  HOSTDEVICE Color(const Color& i_other) = default;
  HOSTDEVICE Color& operator=(const Color& i_other);

  HOSTDEVICE Color operator*(double i_factor) const;
  HOSTDEVICE Color operator*(const Vector3d& i_factor) const;
  HOSTDEVICE Color operator+(const Color& i_other) const;
  HOSTDEVICE Color& operator+=(const Color& i_other);

  bool operator==(const Color& i_other) const;
  bool operator!=(const Color& i_other) const;

  HOSTDEVICE [[nodiscard]] std::uint8_t GetRed() const;
  HOSTDEVICE [[nodiscard]] std::uint8_t GetGreen() const;
  HOSTDEVICE [[nodiscard]] std::uint8_t GetBlue() const;
  HOSTDEVICE [[nodiscard]] std::uint8_t GetAlpha() const;

  void SetRed(std::uint8_t i_red);
  void SetGreen(std::uint8_t i_green);
  void SetBlue(std::uint8_t i_blue);
  void SetAlpha(std::uint8_t i_alpha);

  HOSTDEVICE explicit operator std::uint32_t() const;
  [[nodiscard]] std::uint32_t GetRGBA() const;
  void SetRGBA(std::uint32_t i_rgba);

  [[nodiscard]] std::string Serialize() const;

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
