#pragma once

template<class T>
TransformationMatrix::TransformationMatrix()
  {
  SetIdentity();
  }

template<class T>
void TransformationMatrix::SetIdentity()
  {
  for (auto i = 0u; i < 16; ++i)
    {
    m_matrix[i/4][i%4] = !(i%5);
    }
  m_is_identity = true;
  }

template<class T>
template<class U>
void TransformationMatrix::SetDiagonal(U i_number)
  {
  for (auto i = 0u; i < 16; ++i)
    {
    m_matrix[i / 4][i % 4] = i_number * !(i % 5);
    }
  m_is_identity = false;
  }