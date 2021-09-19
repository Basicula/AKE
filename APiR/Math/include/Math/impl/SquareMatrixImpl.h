#include "..\SquareMatrix.h"
template<class ElementType, size_t N>
SquareMatrix<ElementType, N>::SquareMatrix() {
  SetZero();
}

template<class ElementType, size_t N>
template<std::size_t D, typename T>
SquareMatrix<ElementType, N>::SquareMatrix(
  ElementType i_a00, ElementType i_a01,
  ElementType i_a10, ElementType i_a11) {
  m_matrix[0][0] = i_a00; m_matrix[0][1] = i_a01;
  m_matrix[1][0] = i_a10; m_matrix[1][1] = i_a11;
}

template<class ElementType, size_t N>
template<std::size_t D, typename T>
SquareMatrix<ElementType, N>::SquareMatrix(
  ElementType i_a00, ElementType i_a01, ElementType i_a02,
  ElementType i_a10, ElementType i_a11, ElementType i_a12,
  ElementType i_a20, ElementType i_a21, ElementType i_a22) {
  m_matrix[0][0] = i_a00; m_matrix[0][1] = i_a01; m_matrix[0][2] = i_a02;
  m_matrix[1][0] = i_a10; m_matrix[1][1] = i_a11; m_matrix[1][2] = i_a12;
  m_matrix[2][0] = i_a20; m_matrix[2][1] = i_a21; m_matrix[2][2] = i_a22;
}

template<class ElementType, size_t N>
SquareMatrix<ElementType, N>::SquareMatrix(const SquareMatrix& i_other) {
  std::copy(i_other.m_elements, i_other.m_elements + m_size, m_elements);
}

template<class ElementType, size_t N>
void SquareMatrix<ElementType, N>::SetIdentity() {
  SetZero();
  for (size_t diagonal_id = 0; diagonal_id < m_size; diagonal_id += N + 1)
    m_elements[diagonal_id] = 1;
}

template<class ElementType, size_t N>
void SquareMatrix<ElementType, N>::SetZero() {
  std::fill_n(m_elements, m_size, 0);
}

template<class ElementType, size_t N>
void SquareMatrix<ElementType, N>::Transpose() {
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j)
      std::swap(m_matrix[i][j], m_matrix[j][i]);
}

template<class ElementType, size_t N>
SquareMatrix<ElementType, N> SquareMatrix<ElementType, N>::Transposed() const {
  SquareMatrix<ElementType, N> res(*this);
  res.Transpose();
  return res;
}

template<class ElementType, size_t N>
ElementType SquareMatrix<ElementType, N>::operator()(std::size_t i, std::size_t j) const {
  return m_matrix[i][j];
}

template<class ElementType, size_t N>
ElementType& SquareMatrix<ElementType, N>::operator()(std::size_t i, std::size_t j) {
  return m_matrix[i][j];
}

template<class ElementType, size_t N>
SquareMatrix<ElementType, N> SquareMatrix<ElementType, N>::operator*(const SquareMatrix<ElementType, N>& i_other) const {
  SquareMatrix<ElementType, N> res;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      for (size_t k = 0; k < N; ++k)
        res.m_matrix[i][j] += m_matrix[i][k] * i_other.m_matrix[k][j];
  return res;
}

template<class ElementType, size_t N>
SquareMatrix<ElementType, N>& SquareMatrix<ElementType, N>::operator*=(const SquareMatrix<ElementType, N>& i_other) {
  *this = *this * i_other;
  return *this;
}

template<class ElementType, size_t N>
void SquareMatrix<ElementType, N>::ApplyLeft(Vector<ElementType, N>& io_vector) const {
  Vector<ElementType, N> res;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      res[i] += io_vector[j] * m_matrix[j][i];
  io_vector = res;
}

template<class ElementType, size_t N>
Vector<ElementType, N> SquareMatrix<ElementType, N>::ApplyLeft(const Vector<ElementType, N>& i_vector) const {
  Vector<ElementType, N> res;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      res[i] += i_vector[j] * m_matrix[j][i];
  return res;
}

template<class ElementType, size_t N>
void SquareMatrix<ElementType, N>::ApplyRight(Vector<ElementType, N>& io_vector) const {
  Vector<ElementType, N> res;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      res[i] += io_vector[j] * m_matrix[i][j];
  io_vector = res;
}

template<class ElementType, size_t N>
Vector<ElementType, N> SquareMatrix<ElementType, N>::ApplyRight(const Vector<ElementType, N>& i_vector) const {
  Vector<ElementType, N> res;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      res[i] += i_vector[j] * m_matrix[i][j];
  return res;
}

template<class ElementType, size_t N>
bool SquareMatrix<ElementType, N>::operator==(const SquareMatrix& i_other) const {
  for (size_t i = 0; i < m_size; ++i)
    if (m_elements[i] != i_other.m_elements[i])
      return false;
  return true;
}
