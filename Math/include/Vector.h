#pragma once

template<size_t Dimension = 3, class ElementType = double>
class Vector;

template<size_t Dimension, class ElementType>
class Vector
{
public:
  template<size_t D = 3, typename T = typename ElementType>
  Vector(ElementType i_x, ElementType i_y, ElementType i_z);

  Vector();
  Vector(const Vector& i_other);

  ElementType* GetArray() const;

  Vector operator-(const Vector& i_other) const;
  void operator-=(const Vector& i_other);

  Vector operator+(const Vector& i_other) const;
  void operator+=(const Vector& i_other);

  template<class Factor>
  Vector operator*(Factor i_factor) const;
  template<class Factor>
  void operator*=(Factor i_factor);

  template<size_t D = 3, typename T>
  Vector<D,T> CrossProduct(const Vector<D,T>& i_other) const;
  ElementType Dot(const Vector& i_other) const;
  void Normalize();
  Vector Normalized() const;
  double Length() const;
  ElementType SquareLength() const;
  double Distance(const Vector& i_other) const;

private:
  ElementType m_coords[Dimension];
};

#include <VectorImpl.h>

using Vector3d = Vector<3, double>;