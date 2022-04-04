#pragma once
template <class ElementType>
Matrix3<ElementType>::Matrix3()
{
  SetIdentity();
}

template <class ElementType>
inline Matrix3<ElementType>::Matrix3(const Matrix3& i_other)
{
  std::copy(i_other.m_elements, i_other.m_elements + m_element_cnt, m_elements);
}

template <class ElementType>
void Matrix3<ElementType>::SetIdentity()
{
  std::fill_n(m_elements, m_element_cnt, 0.0);
  m_elements[0] = 1.0;
  m_elements[4] = 1.0;
  m_elements[8] = 1.0;
}

template <class ElementType>
inline Matrix3<ElementType>::Matrix3(const std::initializer_list<ElementType>& i_init_list)
{
  std::fill_n(m_elements, m_element_cnt, 0.0);
  const auto end_it = i_init_list.size() > m_element_cnt ? i_init_list.begin() + m_element_cnt : i_init_list.end();
  std::copy(i_init_list.begin(), end_it, m_elements);
}

template <class ElementType>
inline ElementType Matrix3<ElementType>::operator()(std::size_t i, std::size_t j) const
{
  return m_matrix[i][j];
}

template <class ElementType>
inline ElementType& Matrix3<ElementType>::operator()(std::size_t i, std::size_t j)
{
  return m_matrix[i][j];
}

template <class ElementType>
Vector3d Matrix3<ElementType>::operator*(const Vector3d& i_vector) const
{
  const double x = m_matrix[0][0] * i_vector[0] + m_matrix[0][1] * i_vector[1] + m_matrix[0][2] * i_vector[2];
  const double y = m_matrix[1][0] * i_vector[0] + m_matrix[1][1] * i_vector[1] + m_matrix[1][2] * i_vector[2];
  const double z = m_matrix[2][0] * i_vector[0] + m_matrix[2][1] * i_vector[1] + m_matrix[2][2] * i_vector[2];
  return Vector3d(x, y, z);
}

template <class ElementType>
Matrix3<ElementType> Matrix3<ElementType>::operator*(const Matrix3& i_other) const
{
  Matrix3 temp;
  for (auto i = 0u; i < m_matrix_dim; ++i)
    for (auto j = 0u; j < m_matrix_dim; ++j) {
      temp.m_matrix[i][j] = m_matrix[i][0] * i_other.m_matrix[0][j] + m_matrix[i][1] * i_other.m_matrix[1][j] +
                            m_matrix[i][2] * i_other.m_matrix[2][j];
    }
  return temp;
}

template <class ElementType>
void Matrix3<ElementType>::ApplyLeft(Vector3d& io_vector) const
{
  const double x = m_matrix[0][0] * io_vector[0] + m_matrix[0][1] * io_vector[1] + m_matrix[0][2] * io_vector[2];
  const double y = m_matrix[1][0] * io_vector[0] + m_matrix[1][1] * io_vector[1] + m_matrix[1][2] * io_vector[2];
  const double z = m_matrix[2][0] * io_vector[0] + m_matrix[2][1] * io_vector[1] + m_matrix[2][2] * io_vector[2];

  io_vector[0] = x;
  io_vector[1] = y;
  io_vector[2] = z;
}

template <class ElementType>
void Matrix3<ElementType>::ApplyRight(Vector3d& io_vector) const
{
  const double x = m_elements[0] * io_vector[0] + m_elements[3] * io_vector[1] + m_elements[6] * io_vector[2];
  const double y = m_elements[1] * io_vector[0] + m_elements[4] * io_vector[1] + m_elements[7] * io_vector[2];
  const double z = m_elements[2] * io_vector[0] + m_elements[5] * io_vector[1] + m_elements[8] * io_vector[2];

  io_vector[0] = x;
  io_vector[1] = y;
  io_vector[2] = z;
}

template <class ElementType>
void Matrix3<ElementType>::Transpose()
{
  // 0 1 2
  // 3 4 5
  // 6 7 8
  std::swap(m_matrix[0][1], m_matrix[1][0]);
  std::swap(m_matrix[0][2], m_matrix[2][0]);
  std::swap(m_matrix[1][2], m_matrix[2][1]);
}

template <class ElementType>
Matrix3<ElementType> Matrix3<ElementType>::Transposed() const
{
  Matrix3 res(*this);
  res.Transpose();
  return res;
}
