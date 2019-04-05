#pragma once
#include <TransformationMatrix.h>

using ushort = unsigned short;

TransformationMatrix::TransformationMatrix()
  {
  SetIdentity();
  }

TransformationMatrix::TransformationMatrix(double* i_from16)
  {
  for (ushort i = 0; i < 16; ++i)
    m_matrix[i / 4][i % 4] = i_from16[i];
  m_is_identity = _CheckIdentity();
  }

TransformationMatrix::TransformationMatrix(const TransformationMatrix& i_other)
  {
  if (i_other.IsIdentity())
    SetIdentity();
  else
    {
    for (ushort i = 0; i < 4; ++i)
      for (ushort j = 0; j < 4; ++j)
        {
        m_matrix[i][j] = i_other.m_matrix[i][j];
        m_inverse_matrix[i][j] = i_other.m_inverse_matrix[i][j];
        }
    m_is_identity = false;
    }
  }

void TransformationMatrix::SetIdentity()
  {
  for (ushort i = 0u; i < 16; ++i)
    {
    m_matrix[i / 4][i % 4] = !(i % 5);
    }
  m_is_identity = true;
  }

void TransformationMatrix::SetDiagonal(double i_number)
  {
  for (ushort i = 0u; i < 16; ++i)
    {
    m_matrix[i / 4][i % 4] = i_number * !(i % 5);
    }
  m_is_identity = false;
  }

TransformationMatrix TransformationMatrix::operator*(const TransformationMatrix& i_other) const
  {
  TransformationMatrix res(*this);
  res._Multiply(i_other);
  return res;
  }

void TransformationMatrix::_Multiply(const TransformationMatrix& i_other)
  {
  if (i_other.IsIdentity())
    return;
  if (m_is_identity)
    {
    *this = i_other;
    return;
    }
  for (ushort i = 0; i < 4; ++i)
    for (ushort j = 0; j < 4; ++j)
      {
      m_matrix[i][j] =
        m_matrix[i][0] * i_other.m_matrix[0][j] +
        m_matrix[i][1] * i_other.m_matrix[1][j] +
        m_matrix[i][2] * i_other.m_matrix[2][j] +
        m_matrix[i][3] * i_other.m_matrix[3][j];
      m_inverse_matrix[i][j] =
        m_inverse_matrix[i][0] * i_other.m_inverse_matrix[0][j] +
        m_inverse_matrix[i][1] * i_other.m_inverse_matrix[1][j] +
        m_inverse_matrix[i][2] * i_other.m_inverse_matrix[2][j] +
        m_inverse_matrix[i][3] * i_other.m_inverse_matrix[3][j];
      }
  }

bool TransformationMatrix::_CheckIdentity() const
  {
  bool is_identity = true;
  for(ushort i = 0;i<4;++i)
    for(ushort j = 0;j<4;++j)
      is_identity &= m_matrix[i][j] == (i == j ?  1 : 0);
  return is_identity;
  }