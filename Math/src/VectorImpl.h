#pragma once
#include "Vector.h"
#include <math.h>

template<size_t Dimension, class ElementType>
Vector<Dimension, ElementType>::Vector()
{
  for (size_t i = 0; i < Dimension; ++i)
    m_coords[i] = 0;
};

template<size_t Dimension, class ElementType>
Vector<Dimension, ElementType>::Vector(const Vector& i_other)
{
  for (size_t i = 0; i < Dimension; ++i)
    m_coords[i] = i_other.m_coords[i];
}

template<size_t Dimension, class ElementType>
Vector<Dimension, ElementType> Vector<Dimension, ElementType>::operator-(const Vector<Dimension, ElementType>& i_other) const
{
  Vector<Dimension, ElementType> copy(*this);
  copy -= i_other;
  return copy;
}

template<size_t Dimension, class ElementType>
void Vector<Dimension, ElementType>::operator-=(const Vector<Dimension, ElementType>& i_other)
{
  for (size_t i = 0; i < Dimension; ++i)
    m_coords[i] -= i_other.m_coords[i];
}

template<size_t Dimension, class ElementType>
Vector<Dimension, ElementType> Vector<Dimension, ElementType>::operator+(const Vector<Dimension, ElementType>& i_other) const
{
  Vector<Dimension, ElementType> copy(*this);
  copy += i_other;
  return copy;
}

template<size_t Dimension, class ElementType>
void Vector<Dimension, ElementType>::operator+=(const Vector<Dimension, ElementType>& i_other)
{
  for (size_t i = 0; i < Dimension; ++i)
    m_coords[i] += i_other.m_coords[i];
}

template<size_t Dimension, class ElementType>
template<class Factor>
Vector<Dimension, ElementType> Vector<Dimension, ElementType>::operator*(Factor i_factor) const
{
  Vector<Dimension, ElementType> copy(*this);
  copy *= i_factor;
  return copy;
}

template<size_t Dimension, class ElementType>
template<class Factor>
void Vector<Dimension, ElementType>::operator*=(Factor i_factor)
{
  for (size_t i = 0; i < Dimension; ++i)
    m_coords[i] *= i_factor;
}

template<size_t Dimension, class ElementType>
double Vector<Dimension, ElementType>::Distance(const Vector<Dimension, ElementType>& i_other) const
{
  return (*this-i_other).Length();
}

template<size_t Dimension, class ElementType>
ElementType Vector<Dimension, ElementType>::Dot(const Vector<Dimension, ElementType>& i_other) const
{
  ElementType res = 0;
  for (size_t i = 0; i < Dimension; ++i)
    res += m_coords[i] * i_other.m_coords[i];
  return res;
}

template<size_t Dimension, class ElementType>
void Vector<Dimension, ElementType>::Normalize()
{
  const double length = Length();
  for (size_t i = 0; i < Dimension; ++i)
    m_coords[i] /= length;
}

template<size_t Dimension, class ElementType>
Vector<Dimension, ElementType> Vector<Dimension, ElementType>::Normalized() const
{
  Vector<Dimension, ElementType> temp(*this);
  temp.Normalize();
  return temp;
}

template<size_t Dimension, class ElementType>
double Vector<Dimension, ElementType>::Length() const
{
  return sqrt(SquareLength());
}

template<size_t Dimension, class ElementType>
ElementType Vector<Dimension, ElementType>::SquareLength() const
{
  ElementType res = 0;
  for (size_t i = 0; i < Dimension; ++i)
    res += m_coords[i] * m_coords[i];
  return res;
}
