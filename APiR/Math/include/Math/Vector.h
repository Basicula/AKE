#pragma once
#include "Macros.h"

#include <string>
#include <type_traits>

template <class ElementType, std::size_t Dimension>
class Vector
{
public:
  template <std::size_t D = Dimension, typename T = typename std::enable_if<D == 2>::type>
  HOSTDEVICE Vector(ElementType i_x, ElementType i_y);

  template <std::size_t D = Dimension, typename T = typename std::enable_if<D == 3>::type>
  HOSTDEVICE Vector(ElementType i_x, ElementType i_y, ElementType i_z);

  template <std::size_t D = Dimension, typename T = typename std::enable_if<D == 4>::type>
  HOSTDEVICE Vector(ElementType i_x, ElementType i_y, ElementType i_z, ElementType i_w);

  HOSTDEVICE Vector();
  HOSTDEVICE explicit Vector(ElementType i_elem);
  Vector(const Vector& i_other) = default;

  HOSTDEVICE ElementType& operator[](std::size_t i_index);
  HOSTDEVICE const ElementType& operator[](std::size_t i_index) const;

  bool operator==(const Vector& i_other) const;
  bool operator!=(const Vector& i_other) const;
  bool operator<(const Vector& i_other) const;
  bool operator<=(const Vector& i_other) const;
  bool operator>(const Vector& i_other) const;
  bool operator>=(const Vector& i_other) const;

  HOSTDEVICE Vector operator-() const;
  HOSTDEVICE Vector operator-(const Vector& i_other) const;
  HOSTDEVICE Vector& operator-=(const Vector& i_other);

  HOSTDEVICE Vector operator+(const Vector& i_other) const;
  HOSTDEVICE Vector& operator+=(const Vector& i_other);

  template <class Factor>
  HOSTDEVICE Vector operator*(Factor i_factor) const;
  HOSTDEVICE Vector operator*(const Vector& i_other) const;
  template <class Factor>
  HOSTDEVICE Vector& operator*=(Factor i_factor);
  HOSTDEVICE Vector operator*=(const Vector& i_other);

  template <class Factor>
  HOSTDEVICE Vector operator/(Factor i_factor) const;
  HOSTDEVICE Vector operator/(const Vector& i_other) const;
  template <class Factor>
  HOSTDEVICE Vector& operator/=(Factor i_factor);
  HOSTDEVICE Vector operator/=(const Vector& i_other);

  template <std::size_t D = Dimension, typename T = typename std::enable_if<D == 3>::type>
  HOSTDEVICE [[nodiscard]] Vector<ElementType, Dimension> CrossProduct(
    const Vector<ElementType, Dimension>& i_other) const;
  HOSTDEVICE [[nodiscard]] ElementType Dot(const Vector& i_other) const;
  HOSTDEVICE void Normalize();
  HOSTDEVICE void Invert();
  HOSTDEVICE [[nodiscard]] Vector Normalized() const;
  HOSTDEVICE [[nodiscard]] double Length() const;
  HOSTDEVICE [[nodiscard]] ElementType SquareLength() const;
  HOSTDEVICE [[nodiscard]] double Distance(const Vector& i_other) const;
  HOSTDEVICE [[nodiscard]] ElementType SquareDistance(const Vector& i_other) const;

  [[nodiscard]] std::string Serialize() const;

protected:
private:
  ElementType m_coords[Dimension];

public:
  static constexpr std::size_t m_dimension = Dimension;
  using m_element_type = ElementType;
};

template <class ElementType>
using Vector2 = Vector<ElementType, 2>;
using Vector2d = Vector2<double>;
using Vector2f = Vector2<float>;
using Vector2i = Vector2<int>;

template <class ElementType>
using Vector3 = Vector<ElementType, 3>;
using Vector3d = Vector3<double>;
using Vector3f = Vector3<float>;
using Vector3i = Vector3<int>;

#include "impl/Vector2dImpl.h"
#include "impl/Vector3dImpl.h"
#include "impl/VectorImpl.h"