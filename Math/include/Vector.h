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
    Vector(const Vector& i_other) = default;

    //can throw exception
    ElementType operator[](size_t i_index) const;

    Vector operator-() const;
    Vector operator-(const Vector& i_other) const;
    void operator-=(const Vector& i_other);

    Vector operator+(const Vector& i_other) const;
    void operator+=(const Vector& i_other);

    template<class Factor>
    Vector operator*(Factor i_factor) const;
    template<class Factor>
    void operator*=(Factor i_factor);

    template<size_t D = 3, typename T>
    Vector<D, T> CrossProduct(const Vector<D, T>& i_other) const;
    ElementType Dot(const Vector& i_other) const;
    void Normalize();
    Vector Normalized() const;
    double Length() const;
    ElementType SquareLength() const;
    double Distance(const Vector& i_other) const;
    ElementType SquareDistance(const Vector& i_other) const;

  protected:

  private:
    ElementType m_coords[Dimension];
  };

#include <VectorImpl.h>

using Vector3d = Vector<3, double>;

template<>
inline Vector3d Vector3d::operator-(const Vector3d& i_other) const
  {
  return Vector3d(m_coords[0] - i_other.m_coords[0], m_coords[1] - i_other.m_coords[1], m_coords[2] - i_other.m_coords[2]);
  }

template<>
inline Vector3d Vector3d::operator+(const Vector3d& i_other) const
  {
  return Vector3d(m_coords[0] + i_other.m_coords[0], m_coords[1] + i_other.m_coords[1], m_coords[2] + i_other.m_coords[2]);
  }

template<>
inline double Vector3d::Dot(const Vector3d& i_other) const
  {
  return m_coords[0] * i_other.m_coords[0] + m_coords[1] * i_other.m_coords[1] + m_coords[2] * i_other.m_coords[2];
  }

template<>
template<>
inline Vector3d Vector3d::operator*<double>(double i_val) const
  {
  return Vector3d(m_coords[0]*i_val, m_coords[1] * i_val, m_coords[2] * i_val);
  }

template<>
template<>
inline void Vector3d::operator*=<double>(double i_val)
  {
  m_coords[0] *= i_val;
  m_coords[1] *= i_val;
  m_coords[2] *= i_val;
  }

template<>
inline double Vector3d::SquareLength() const
  {
  return m_coords[0] * m_coords[0] + m_coords[1] * m_coords[1] + m_coords[2] * m_coords[2];
  }

template<>
inline double Vector3d::SquareDistance(const Vector3d& i_other) const
  {
  return (i_other - *this).SquareLength();
  }