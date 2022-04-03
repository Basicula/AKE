#pragma once
#include "Math/Vector.h"

template <class ElementType>
class Matrix3
{
public:
  // identity matrix
  Matrix3();
  Matrix3(const Matrix3& i_other);
  Matrix3(const std::initializer_list<ElementType>& i_init_list);

  void SetIdentity();

  ElementType operator()(std::size_t i, std::size_t j) const;
  ElementType& operator()(std::size_t i, std::size_t j);

  Vector3d operator*(const Vector3d& i_vector) const;
  Matrix3 operator*(const Matrix3& i_other) const;

  // matrix * vector
  void ApplyLeft(Vector3d& io_vector) const;
  // vector * matrix
  void ApplyRight(Vector3d& io_vector) const;

  void Transpose();
  Matrix3 Transposed() const;

private:
  union
  {
    ElementType m_matrix[3][3];
    ElementType m_elements[9];
  };
  static constexpr std::size_t m_element_cnt = 9;
  static constexpr std::size_t m_matrix_dim = 3;
};

using Matrix3d = Matrix3<double>;

#include "Math/impl/Matrix3Impl.h"