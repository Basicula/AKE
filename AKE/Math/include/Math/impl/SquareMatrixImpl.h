#pragma once

template <class ElementType, size_t N>
SquareMatrix<ElementType, N>::SquareMatrix()
{
  SetZero();
}

template <class ElementType, size_t N>
template <std::size_t D, typename T>
SquareMatrix<ElementType, N>::SquareMatrix(ElementType i_a00, ElementType i_a01, ElementType i_a10, ElementType i_a11)
{
  m_data[0][0] = i_a00;
  m_data[0][1] = i_a01;
  m_data[1][0] = i_a10;
  m_data[1][1] = i_a11;
}

template <class ElementType, size_t N>
template <std::size_t D, typename T>
SquareMatrix<ElementType, N>::SquareMatrix(ElementType i_a00,
                                           ElementType i_a01,
                                           ElementType i_a02,
                                           ElementType i_a10,
                                           ElementType i_a11,
                                           ElementType i_a12,
                                           ElementType i_a20,
                                           ElementType i_a21,
                                           ElementType i_a22)
{
  m_data[0][0] = i_a00;
  m_data[0][1] = i_a01;
  m_data[0][2] = i_a02;
  m_data[1][0] = i_a10;
  m_data[1][1] = i_a11;
  m_data[1][2] = i_a12;
  m_data[2][0] = i_a20;
  m_data[2][1] = i_a21;
  m_data[2][2] = i_a22;
}

template <class ElementType, size_t N>
SquareMatrix<ElementType, N>::SquareMatrix(const SquareMatrix& i_other)
{
  std::copy(&i_other.m_data[0][0], &i_other.m_data[0][0] + m_size, &m_data[0][0]);
}

template <class ElementType, size_t N>
void SquareMatrix<ElementType, N>::SetIdentity()
{
  SetZero();
  for (size_t diagonal_id = 0; diagonal_id < N; ++diagonal_id)
    m_data[diagonal_id][diagonal_id] = 1;
}

template <class ElementType, size_t N>
void SquareMatrix<ElementType, N>::SetZero()
{
  std::fill_n(&m_data[0][0], m_size, static_cast<ElementType>(0));
}

template <class ElementType, size_t N>
void SquareMatrix<ElementType, N>::Transpose()
{
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j)
      std::swap(m_data[i][j], m_data[j][i]);
}

template <class ElementType, size_t N>
SquareMatrix<ElementType, N> SquareMatrix<ElementType, N>::Transposed() const
{
  SquareMatrix<ElementType, N> res(*this);
  res.Transpose();
  return res;
}

template <class ElementType, size_t N>
ElementType SquareMatrix<ElementType, N>::operator()(std::size_t i, std::size_t j) const
{
  return m_data[i][j];
}

template <class ElementType, size_t N>
ElementType& SquareMatrix<ElementType, N>::operator()(std::size_t i, std::size_t j)
{
  return m_data[i][j];
}

template <class ElementType, size_t N>
SquareMatrix<ElementType, N> SquareMatrix<ElementType, N>::operator*(const SquareMatrix<ElementType, N>& i_other) const
{
  SquareMatrix<ElementType, N> res;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      for (size_t k = 0; k < N; ++k)
        res.m_data[i][j] += m_data[i][k] * i_other.m_data[k][j];
  return res;
}

template <class ElementType, size_t N>
SquareMatrix<ElementType, N>& SquareMatrix<ElementType, N>::operator*=(const SquareMatrix<ElementType, N>& i_other)
{
  *this = *this * i_other;
  return *this;
}

template <class ElementType, size_t N>
void SquareMatrix<ElementType, N>::ApplyLeft(Vector<ElementType, N>& io_vector) const
{
  Vector<ElementType, N> res;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      res[i] += io_vector[j] * m_data[j][i];
  io_vector = res;
}

template <class ElementType, size_t N>
Vector<ElementType, N> SquareMatrix<ElementType, N>::ApplyLeft(const Vector<ElementType, N>& i_vector) const
{
  Vector<ElementType, N> res;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      res[i] += i_vector[j] * m_data[j][i];
  return res;
}

template <class ElementType, size_t N>
void SquareMatrix<ElementType, N>::ApplyRight(Vector<ElementType, N>& io_vector) const
{
  Vector<ElementType, N> res;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      res[i] += io_vector[j] * m_data[i][j];
  io_vector = res;
}

template <class ElementType, size_t N>
Vector<ElementType, N> SquareMatrix<ElementType, N>::ApplyRight(const Vector<ElementType, N>& i_vector) const
{
  Vector<ElementType, N> res;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      res[i] += i_vector[j] * m_data[i][j];
  return res;
}

template <class ElementType, size_t N>
bool SquareMatrix<ElementType, N>::operator==(const SquareMatrix& i_other) const
{
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      if (m_data[i][j] != i_other.m_data[i][j])
        return false;
  return true;
}
