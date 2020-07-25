#include "..\Matrix3.h"
#pragma once

template<class ElementType>
Matrix3<ElementType>::Matrix3()
  {
  SetIdentity();
  }

template<class ElementType>
inline Matrix3<ElementType>::Matrix3(const Matrix3& i_other)
  {
  std::copy(i_other.m_matrix, i_other.m_matrix + m_element_cnt, m_matrix);
  }

template<class ElementType>
void Matrix3<ElementType>::SetIdentity()
  {
  std::fill_n(m_matrix, m_element_cnt, 0.0);
  m_matrix[0] = 1.0;
  m_matrix[4] = 1.0;
  m_matrix[8] = 1.0;
  }

template<class ElementType>
inline Matrix3<ElementType>::Matrix3(const std::initializer_list<ElementType>& i_init_list)
  {
  std::fill_n(m_matrix, m_element_cnt, 0.0);
  const auto end_it = i_init_list.size() > m_element_cnt ?
    i_init_list.begin() + m_element_cnt :
    i_init_list.end();
  std::copy(i_init_list.begin(), end_it, m_matrix);
  }

template<class ElementType>
inline ElementType Matrix3<ElementType>::operator()(std::size_t i, std::size_t j) const
  {
  return m_matrix[i * 3 + j];
  }

template<class ElementType>
inline ElementType& Matrix3<ElementType>::operator()(std::size_t i, std::size_t j)
  {
  return m_matrix[i * 3 + j];
  }

template<class ElementType>
Vector3d Matrix3<ElementType>::operator*(const Vector3d& i_vector) const
  {
  const double x =
    m_matrix[0] * i_vector[0] + 
    m_matrix[1] * i_vector[1] + 
    m_matrix[2] * i_vector[2];
  const double y =
    m_matrix[3] * i_vector[0] + 
    m_matrix[4] * i_vector[1] + 
    m_matrix[5] * i_vector[2];
  const double z =
    m_matrix[6] * i_vector[0] + 
    m_matrix[7] * i_vector[1] + 
    m_matrix[8] * i_vector[2];
  return Vector3d(x, y, z);
  }

template<class ElementType>
Matrix3<ElementType> Matrix3<ElementType>::operator*(const Matrix3& i_other) const
  {
  Matrix3 temp;
  for (auto i = 0u; i < m_element_cnt; ++i)
    {
    const auto row = m_matrix_dim * (i / m_matrix_dim);
    const auto col = i % m_matrix_dim;
    temp.m_matrix[i] =
      m_matrix[row + 0] * i_other.m_matrix[0 + col] +
      m_matrix[row + 1] * i_other.m_matrix[3 + col] +
      m_matrix[row + 2] * i_other.m_matrix[6 + col];
    }
  return temp;
  }

template<class ElementType>
void Matrix3<ElementType>::Transpose()
  {
  // 0 1 2
  // 3 4 5
  // 6 7 8
  std::swap(m_matrix[1], m_matrix[3]);
  std::swap(m_matrix[2], m_matrix[6]);
  std::swap(m_matrix[5], m_matrix[7]);
  }

template<class ElementType>
Matrix3<ElementType> Matrix3<ElementType>::Transposed() const
  {
  Matrix3 res(*this);
  res.Transpose();
  return res;
  }
