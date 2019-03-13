#pragma once

class Color
{
public:
  Color();
  Color(unsigned char i_red, unsigned char i_green, unsigned char i_blue);

  /*template<class T>
  Color(T i_red, T i_green, T i_blue);*/

  Color(const Color& i_other);

  bool operator==(const Color& i_other) const;
  unsigned char* RGB() const;
  unsigned char* BGR() const;
private:
  unsigned char m_red;
  unsigned char m_green;
  unsigned char m_blue;
};