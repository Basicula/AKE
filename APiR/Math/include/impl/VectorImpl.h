#pragma once
#include <math.h>
#include <algorithm>
#include <stdexcept>
#include <Vector.h>

template<std::size_t Dimension, class ElementType>
Vector<Dimension, ElementType>::Vector()
  : Vector(0)
  {
  };
  
template<std::size_t Dimension, class ElementType>
Vector<Dimension, ElementType>::Vector(ElementType i_elem)
  {
  for (std::size_t i = 0; i < Dimension; ++i)
    m_coords[i] = i_elem;
  };
  
template<std::size_t Dimension, class ElementType>
template<std::size_t D, typename T>
inline Vector<Dimension, ElementType>::Vector(ElementType i_x, ElementType i_y)
  : m_coords{i_x, i_y}
  {}

template<std::size_t Dimension, class ElementType>
template<std::size_t D, typename T>
inline Vector<Dimension, ElementType>::Vector(
  ElementType i_x, 
  ElementType i_y, 
  ElementType i_z)
  : m_coords{i_x, i_y, i_z}
  {}
  
template<std::size_t Dimension, class ElementType>
template<std::size_t D, typename T>
inline Vector<Dimension, ElementType>::Vector(
  ElementType i_x, 
  ElementType i_y, 
  ElementType i_z, 
  ElementType i_w)
  : m_coords{i_x, i_y, i_z, i_w}
  {}

template<std::size_t Dimension, class ElementType>
inline ElementType& Vector<Dimension, ElementType>::operator[](std::size_t i_index)
  {
  return m_coords[i_index];
  }

template<std::size_t Dimension, class ElementType>
inline const ElementType& 
Vector<Dimension, ElementType>::operator[](std::size_t i_index) const
  {
  return m_coords[i_index];
  }

template<std::size_t Dimension, class ElementType>
bool Vector<Dimension, ElementType>::operator==(const Vector<Dimension, ElementType>& i_other) const
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
  
template<std::size_t Dimension, class ElementType>
bool Vector<Dimension, ElementType>::operator<(const Vector<Dimension, ElementType>& i_other) const
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

template<std::size_t Dimension, class ElementType>
bool Vector<Dimension, ElementType>::operator<=(const Vector<Dimension, ElementType>& i_other) const
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

template<std::size_t Dimension, class ElementType>
bool Vector<Dimension, ElementType>::operator>(const Vector<Dimension, ElementType>& i_other) const
  {
  return !(*this <= i_other);
  }

template<std::size_t Dimension, class ElementType>
bool Vector<Dimension, ElementType>::operator>=(const Vector<Dimension, ElementType>& i_other) const
  {
  return !(*this < i_other);
  }

template<std::size_t Dimension, class ElementType>
bool Vector<Dimension, ElementType>::operator!=(const Vector<Dimension, ElementType>& i_other) const
  {
  return !(*this == i_other);
  }

template<std::size_t Dimension, class ElementType>
Vector<Dimension, ElementType> Vector<Dimension, ElementType>::operator-() const
  {
  Vector<Dimension,ElementType> copy(*this);
  for(auto& coord : copy.m_coords)
    coord = -coord;
  return copy;
  }

template<std::size_t Dimension, class ElementType>
Vector<Dimension, ElementType> Vector<Dimension, ElementType>::operator-(const Vector<Dimension, ElementType>& i_other) const
  {
  Vector<Dimension, ElementType> copy(*this);
  copy -= i_other;
  return copy;
  }

template<std::size_t Dimension, class ElementType>
Vector<Dimension, ElementType>& Vector<Dimension, ElementType>::operator-=(const Vector<Dimension, ElementType>& i_other)
  {
  for (std::size_t i = 0; i < Dimension; ++i)
    m_coords[i] -= i_other.m_coords[i];
  return *this;
  }

template<std::size_t Dimension, class ElementType>
Vector<Dimension, ElementType> Vector<Dimension, ElementType>::operator+(const Vector<Dimension, ElementType>& i_other) const
  {
  Vector<Dimension, ElementType> copy(*this);
  copy += i_other;
  return copy;
  }

template<std::size_t Dimension, class ElementType>
Vector<Dimension, ElementType>& Vector<Dimension, ElementType>::operator+=(const Vector<Dimension, ElementType>& i_other)
  {
  for (std::size_t i = 0; i < Dimension; ++i)
    m_coords[i] += i_other.m_coords[i];
  return *this;
  }

template<std::size_t Dimension, class ElementType>
template<class Factor>
Vector<Dimension, ElementType> Vector<Dimension, ElementType>::operator*(Factor i_factor) const
  {
  Vector<Dimension, ElementType> copy(*this);
  copy *= i_factor;
  return copy;
  }

template<std::size_t Dimension, class ElementType>
template<class Factor>
Vector<Dimension, ElementType>& Vector<Dimension, ElementType>::operator*=(Factor i_factor)
  {
  for (std::size_t i = 0; i < Dimension; ++i)
    m_coords[i] *= i_factor;
  return *this;
  }
  
template<std::size_t Dimension, class ElementType>
template<class Factor>
Vector<Dimension, ElementType> Vector<Dimension, ElementType>::operator/(Factor i_factor) const
  {
  Vector<Dimension, ElementType> copy(*this);
  copy /= i_factor;
  return copy;
  }

template<std::size_t Dimension, class ElementType>
template<class Factor>
Vector<Dimension, ElementType>& Vector<Dimension, ElementType>::operator/=(Factor i_factor)
  {
  for (std::size_t i = 0; i < Dimension; ++i)
    m_coords[i] /= i_factor;
  return *this;
  }

template<std::size_t Dimension, class ElementType>
double Vector<Dimension, ElementType>::Distance(const Vector<Dimension, ElementType>& i_other) const
  {
  return (*this - i_other).Length();
  }

template<std::size_t Dimension, class ElementType>
inline ElementType Vector<Dimension, ElementType>::SquareDistance(const Vector & i_other) const
  {
  return (*this - i_other).SquareLength();
  }

template<std::size_t Dimension, class ElementType>
template<std::size_t D, typename T>
Vector<Dimension, ElementType> Vector<Dimension, ElementType>::CrossProduct(const Vector<Dimension, ElementType>& i_other) const
  {
  Vector<Dimension, ElementType> res;
  res.m_coords[0] = m_coords[1] * i_other.m_coords[2] - m_coords[2] * i_other.m_coords[1];
  res.m_coords[1] = m_coords[2] * i_other.m_coords[0] - m_coords[0] * i_other.m_coords[2];
  res.m_coords[2] = m_coords[0] * i_other.m_coords[1] - m_coords[1] * i_other.m_coords[0];
  return res;
  }

template<std::size_t Dimension, class ElementType>
ElementType Vector<Dimension, ElementType>::Dot(const Vector<Dimension, ElementType>& i_other) const
  {
  ElementType res = 0;
  for (std::size_t i = 0; i < Dimension; ++i)
    res += m_coords[i] * i_other.m_coords[i];
  return res;
  }

template<std::size_t Dimension, class ElementType>
void Vector<Dimension, ElementType>::Normalize()
  {
  const double length = Length();
  if (length > 0.0)
    for (std::size_t i = 0; i < Dimension; ++i)
      m_coords[i] /= length;
  }

template<std::size_t Dimension, class ElementType>
Vector<Dimension, ElementType> Vector<Dimension, ElementType>::Normalized() const
  {
  Vector<Dimension, ElementType> res(*this);
  res.Normalize();
  return res;
  }

template<std::size_t Dimension, class ElementType>
double Vector<Dimension, ElementType>::Length() const
  {
  return sqrt(SquareLength());
  }

template<std::size_t Dimension, class ElementType>
ElementType Vector<Dimension, ElementType>::SquareLength() const
  {
  ElementType res = 0;
  for (std::size_t i = 0; i < Dimension; ++i)
    res += m_coords[i] * m_coords[i];
  return res;
  }

template<std::size_t Dimension, class ElementType>
std::string Vector<Dimension, ElementType>::Serialize() const
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