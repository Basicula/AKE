#pragma once
#include <algorithm>

#include <Vector.h>

class Color
{
public:
  Color();
  Color(unsigned char i_red, unsigned char i_green, unsigned char i_blue);

  template<class T1, class T2, class T3>
  Color(T1 i_red, T2 i_green, T3 i_blue);

  Color(const Color& i_other);

  Color operator*(double i_factor) const;
  Color operator*(const Vector3d& i_factor) const;
  Color operator+(const Color& i_other) const;

  bool operator==(const Color& i_other) const;
  unsigned char* RGB() const;
  unsigned char* BGR() const;
private:
  unsigned char m_red;
  unsigned char m_green;
  unsigned char m_blue;
};

template<class T1,class T2, class T3>
Color::Color(T1 i_red, T2 i_green, T3 i_blue)
  : m_red(static_cast<unsigned char>(std::max(T1(0), std::min(i_red, T1(255)))))
  , m_green(static_cast<unsigned char>(std::max(T2(0), std::min(i_green, T2(255)))))
  , m_blue(static_cast<unsigned char>(std::max(T3(0), std::min(i_blue, T3(255)))))
  {};