#pragma once

#include <type_traits>
#include <string>
#include <array>

template<std::size_t Dimension, class ElementType>
class Vector
  {
  public:
    template<std::size_t D = Dimension, 
      typename T = typename std::enable_if<D == 2>::type >
    Vector(ElementType i_x, ElementType i_y);
  
    template<std::size_t D = Dimension, 
      typename T = typename std::enable_if<D == 3>::type >
    Vector(ElementType i_x, ElementType i_y, ElementType i_z);
    
    template<std::size_t D = Dimension, 
      typename T = typename std::enable_if<D == 4>::type >
    Vector(ElementType i_x, ElementType i_y, ElementType i_z, ElementType i_w);

    Vector();
    Vector(ElementType i_elem);
    Vector(const Vector& i_other) = default;

    ElementType& operator[](std::size_t i_index);
    const ElementType& operator[](std::size_t i_index) const;
    
    bool operator==(const Vector& i_other) const;
    bool operator!=(const Vector& i_other) const;
    bool operator<(const Vector& i_other) const;
    bool operator<=(const Vector& i_other) const;
    bool operator>(const Vector& i_other) const;
    bool operator>=(const Vector& i_other) const;

    Vector operator-() const;
    Vector operator-(const Vector& i_other) const;
    Vector& operator-=(const Vector& i_other);

    Vector operator+(const Vector& i_other) const;
    Vector& operator+=(const Vector& i_other);

    template<class Factor>
    Vector operator*(Factor i_factor) const;
    template<class Factor>
    Vector& operator*=(Factor i_factor);
    
    template<class Factor>
    Vector operator/(Factor i_factor) const;
    template<class Factor>
    Vector& operator/=(Factor i_factor);

    template<std::size_t D = Dimension, 
      typename T = typename std::enable_if<D == 3>::type >
    Vector<Dimension, ElementType> CrossProduct(
      const Vector<Dimension, 
      ElementType>& i_other) const;
    ElementType Dot(const Vector& i_other) const;
    void Normalize();
    Vector Normalized() const;
    double Length() const;
    ElementType SquareLength() const;
    double Distance(const Vector& i_other) const;
    ElementType SquareDistance(const Vector& i_other) const;

    std::string Serialize() const;
  protected:

  private:
    ElementType m_coords[Dimension];

  public:
    static const std::size_t m_dimension = Dimension;
    using m_element_type = ElementType;
  };

using Vector3d = Vector<3, double>;

#include <Vector3dImpl.h>
#include <VectorImpl.h>