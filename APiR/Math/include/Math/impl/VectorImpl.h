#pragma once
#include <math.h>
#include <algorithm>
#include <stdexcept>
#include <Math/Vector.h>

template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension>::Vector()
  : Vector(0)
  {
  };
  
template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension>::Vector(ElementType i_elem)
  {
  std::fill_n(m_coords, m_dimension, i_elem);
  }

template<class ElementType, std::size_t Dimension>
template<std::size_t D, typename T>
inline Vector<ElementType, Dimension>::Vector(ElementType i_x, ElementType i_y)
  : m_coords{i_x, i_y}
  {}

template<class ElementType, std::size_t Dimension>
template<std::size_t D, typename T>
inline Vector<ElementType, Dimension>::Vector(
  ElementType i_x, 
  ElementType i_y, 
  ElementType i_z)
  : m_coords{i_x, i_y, i_z}
  {}
  
template<class ElementType, std::size_t Dimension>
template<std::size_t D, typename T>
inline Vector<ElementType, Dimension>::Vector(
  ElementType i_x, 
  ElementType i_y, 
  ElementType i_z, 
  ElementType i_w)
  : m_coords{i_x, i_y, i_z, i_w}
  {}

template<class ElementType, std::size_t Dimension>
inline ElementType& Vector<ElementType, Dimension>::operator[](std::size_t i_index)
  {
  return m_coords[i_index];
  }

template<class ElementType, std::size_t Dimension>
inline const ElementType& 
Vector<ElementType, Dimension>::operator[](std::size_t i_index) const
  {
  return m_coords[i_index];
  }

template<class ElementType, std::size_t Dimension>
bool Vector<ElementType, Dimension>::operator==(const Vector<ElementType, Dimension>& i_other) const
  {
  bool equal = true;
  for (auto i = 0u; i < Dimension; ++i)
    {
    equal &= (m_coords[i] == i_other.m_coords[i]);
    if (!equal)
      return false;
    }
  return true;
  }
  
template<class ElementType, std::size_t Dimension>
bool Vector<ElementType, Dimension>::operator<(const Vector<ElementType, Dimension>& i_other) const
  {
  bool res = true;
  bool eq = true;
  for (auto i = 0u; i < Dimension; ++i)
    {
    res &= (m_coords[i] <= i_other.m_coords[i]);
    eq &= (m_coords[i] == i_other.m_coords[i]);
    if (!res)
      return false;
    }
  return !eq;
  }

template<class ElementType, std::size_t Dimension>
bool Vector<ElementType, Dimension>::operator<=(const Vector<ElementType, Dimension>& i_other) const
  {
  bool res = true;
  for (auto i = 0u; i < Dimension; ++i)
    {
    res &= (m_coords[i] <= i_other.m_coords[i]);
    if (!res)
      return false;
    }
  return true;
  }

template<class ElementType, std::size_t Dimension>
bool Vector<ElementType, Dimension>::operator>(const Vector<ElementType, Dimension>& i_other) const
  {
  return !(*this <= i_other);
  }

template<class ElementType, std::size_t Dimension>
bool Vector<ElementType, Dimension>::operator>=(const Vector<ElementType, Dimension>& i_other) const
  {
  return !(*this < i_other);
  }

template<class ElementType, std::size_t Dimension>
bool Vector<ElementType, Dimension>::operator!=(const Vector<ElementType, Dimension>& i_other) const
  {
  return !(*this == i_other);
  }

template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension> Vector<ElementType, Dimension>::operator-() const
  {
  Vector<ElementType, Dimension> copy(*this);
  for(auto& coord : copy.m_coords)
    coord = -coord;
  return copy;
  }

template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension> Vector<ElementType, Dimension>::operator-(const Vector<ElementType, Dimension>& i_other) const
  {
  Vector<ElementType, Dimension> copy(*this);
  copy -= i_other;
  return copy;
  }

template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension>& Vector<ElementType, Dimension>::operator-=(const Vector<ElementType, Dimension>& i_other)
  {
  for (std::size_t i = 0; i < Dimension; ++i)
    m_coords[i] -= i_other.m_coords[i];
  return *this;
  }

template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension> Vector<ElementType, Dimension>::operator+(const Vector<ElementType, Dimension>& i_other) const
  {
  Vector<ElementType, Dimension> copy(*this);
  copy += i_other;
  return copy;
  }

template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension>& Vector<ElementType, Dimension>::operator+=(const Vector<ElementType, Dimension>& i_other)
  {
  for (std::size_t i = 0; i < Dimension; ++i)
    m_coords[i] += i_other.m_coords[i];
  return *this;
  }

template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension> Vector<ElementType, Dimension>::operator*(const Vector& i_other) const
  {
  Vector<ElementType, Dimension> copy(*this);
  copy *= i_other;
  return copy;
  }

template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension> Vector<ElementType, Dimension>::operator*=(const Vector& i_other)
  {
  for (std::size_t i = 0; i < Dimension; ++i)
    m_coords[i] *= i_other.m_coords[i];
  return *this;
  }

template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension> Vector<ElementType, Dimension>::operator/(const Vector& i_other) const
  {
  Vector<ElementType, Dimension> copy(*this);
  copy /= i_other;
  return copy;
  }

template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension> Vector<ElementType, Dimension>::operator/=(const Vector& i_other)
  {
  for (std::size_t i = 0; i < Dimension; ++i)
    m_coords[i] /= i_other.m_coords[i];
  return *this;
  }

template<class ElementType, std::size_t Dimension>
template<class Factor>
Vector<ElementType, Dimension> Vector<ElementType, Dimension>::operator*(Factor i_factor) const
  {
  Vector<ElementType, Dimension> copy(*this);
  copy *= i_factor;
  return copy;
  }

template<class ElementType, std::size_t Dimension>
template<class Factor>
Vector<ElementType, Dimension>& Vector<ElementType, Dimension>::operator*=(Factor i_factor)
  {
  for (std::size_t i = 0; i < Dimension; ++i)
    m_coords[i] *= i_factor;
  return *this;
  }
  
template<class ElementType, std::size_t Dimension>
template<class Factor>
Vector<ElementType, Dimension> Vector<ElementType, Dimension>::operator/(Factor i_factor) const
  {
  Vector<ElementType, Dimension> copy(*this);
  copy /= i_factor;
  return copy;
  }

template<class ElementType, std::size_t Dimension>
template<class Factor>
Vector<ElementType, Dimension>& Vector<ElementType, Dimension>::operator/=(Factor i_factor)
  {
  for (std::size_t i = 0; i < Dimension; ++i)
    m_coords[i] /= i_factor;
  return *this;
  }

template<class ElementType, std::size_t Dimension>
double Vector<ElementType, Dimension>::Distance(const Vector<ElementType, Dimension>& i_other) const
  {
  return (*this - i_other).Length();
  }

template<class ElementType, std::size_t Dimension>
inline ElementType Vector<ElementType, Dimension>::SquareDistance(const Vector & i_other) const
  {
  return (*this - i_other).SquareLength();
  }

template<class ElementType, std::size_t Dimension>
template<std::size_t D, typename T>
Vector<ElementType, Dimension> Vector<ElementType, Dimension>::CrossProduct(const Vector<ElementType, Dimension>& i_other) const
  {
  Vector<ElementType, Dimension> res;
  res.m_coords[0] = m_coords[1] * i_other.m_coords[2] - m_coords[2] * i_other.m_coords[1];
  res.m_coords[1] = m_coords[2] * i_other.m_coords[0] - m_coords[0] * i_other.m_coords[2];
  res.m_coords[2] = m_coords[0] * i_other.m_coords[1] - m_coords[1] * i_other.m_coords[0];
  return res;
  }

template<class ElementType, std::size_t Dimension>
ElementType Vector<ElementType, Dimension>::Dot(const Vector<ElementType, Dimension>& i_other) const
  {
  ElementType res = 0;
  for (std::size_t i = 0; i < Dimension; ++i)
    res += m_coords[i] * i_other.m_coords[i];
  return res;
  }

template<class ElementType, std::size_t Dimension>
void Vector<ElementType, Dimension>::Normalize()
  {
  const double length = Length();
  if (length > 0.0)
    for (std::size_t i = 0; i < Dimension; ++i)
      m_coords[i] /= length;
  }

template<class ElementType, std::size_t Dimension>
Vector<ElementType, Dimension> Vector<ElementType, Dimension>::Normalized() const
  {
  Vector<ElementType, Dimension> res(*this);
  res.Normalize();
  return res;
  }

template<class ElementType, std::size_t Dimension>
double Vector<ElementType, Dimension>::Length() const
  {
  return sqrt(SquareLength());
  }

template<class ElementType, std::size_t Dimension>
inline ElementType Vector<ElementType, Dimension>::SquareLength() const
  {
  ElementType res = 0;
  for (std::size_t i = 0; i < Dimension; ++i)
    res += m_coords[i] * m_coords[i];
  return res;
  }

template<class ElementType, std::size_t Dimension>
std::string Vector<ElementType, Dimension>::Serialize() const
  {
  std::string res = "{ \"Vector" + std::to_string(m_dimension) + "d\" : [";
  for (auto i = 0u; i < m_dimension; ++i)
    {
    res += std::to_string(m_coords[i]);
    res += (i == m_dimension-1 ? "" : ", ");
    }
  res += "] }";
  return res;
  }