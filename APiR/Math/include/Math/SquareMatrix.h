#pragma once
#include <Math/Vector.h>
#include <type_traits>

template <class ElementType, size_t N>
class SquareMatrix
{
  static_assert(N > 1, "Bad sizes for matrix");

public:
  // identity matrix
  SquareMatrix();
  SquareMatrix(const SquareMatrix& i_other);

  template <std::size_t D = N, typename T = typename std::enable_if<D == 2>::type>
  SquareMatrix(ElementType i_a00, ElementType i_a01, ElementType i_a10, ElementType i_a11);
  template <std::size_t D = N, typename T = typename std::enable_if<D == 3>::type>
  SquareMatrix(ElementType i_a00,
               ElementType i_a01,
               ElementType i_a02,
               ElementType i_a10,
               ElementType i_a11,
               ElementType i_a12,
               ElementType i_a20,
               ElementType i_a21,
               ElementType i_a22);

  void SetIdentity();
  void SetZero();

  void Transpose();
  SquareMatrix Transposed() const;

  ElementType operator()(std::size_t i, std::size_t j) const;
  ElementType& operator()(std::size_t i, std::size_t j);

  SquareMatrix operator*(const SquareMatrix& i_other) const;
  SquareMatrix& operator*=(const SquareMatrix& i_other);

  // vector = matrix * vector
  void ApplyLeft(Vector<ElementType, N>& io_vector) const;
  Vector<ElementType, N> ApplyLeft(const Vector<ElementType, N>& i_vector) const;

  // vector = vector * matrix
  void ApplyRight(Vector<ElementType, N>& io_vector) const;
  Vector<ElementType, N> ApplyRight(const Vector<ElementType, N>& i_vector) const;

  bool operator==(const SquareMatrix& i_other) const;

protected:
  static constexpr size_t m_size = N * N;
  ElementType m_data[N][N];
};

template <class ElementType>
using Matrix2x2 = SquareMatrix<ElementType, 2>;
using Matrix2x2d = Matrix2x2<double>;
using Matrix2x2f = Matrix2x2<float>;
using Matrix2x2i = Matrix2x2<int>;

template <class ElementType>
using Matrix3x3 = SquareMatrix<ElementType, 3>;
using Matrix3x3d = Matrix3x3<double>;
using Matrix3x3f = Matrix3x3<float>;
using Matrix3x3i = Matrix3x3<int>;

#include "impl/SquareMatrixImpl.h"