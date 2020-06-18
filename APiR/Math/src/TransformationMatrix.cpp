#include <TransformationMatrix.h>

using ushort = unsigned short;

TransformationMatrix::TransformationMatrix()
  {
  SetIdentity();
  }

TransformationMatrix::TransformationMatrix(double* i_from16)
  {
  m_matrix[0][0] = i_from16[0];   m_matrix[0][1] = i_from16[1];   m_matrix[0][2] = i_from16[2];   m_matrix[0][3] = i_from16[3];
  m_matrix[1][0] = i_from16[4];   m_matrix[1][1] = i_from16[5];   m_matrix[1][2] = i_from16[6];   m_matrix[1][3] = i_from16[7];
  m_matrix[2][0] = i_from16[8];   m_matrix[2][1] = i_from16[9];   m_matrix[2][2] = i_from16[10];  m_matrix[2][3] = i_from16[11];
  m_matrix[3][0] = i_from16[12];  m_matrix[3][1] = i_from16[13];  m_matrix[3][2] = i_from16[14];  m_matrix[3][3] = i_from16[15];
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
  SetDiagonal(1);
  m_is_identity = true;
  }

void TransformationMatrix::SetDiagonal(double i_number)
  {
  m_matrix[0][0] = i_number;  m_matrix[0][1] = 0;         m_matrix[0][2] = 0;         m_matrix[0][3] = 0;
  m_matrix[1][0] = 0;         m_matrix[1][1] = i_number;  m_matrix[1][2] = 0;         m_matrix[1][3] = 0;
  m_matrix[2][0] = 0;         m_matrix[2][1] = 0;         m_matrix[2][2] = i_number;  m_matrix[2][3] = 0;
  m_matrix[3][0] = 0;         m_matrix[3][1] = 0;         m_matrix[3][2] = 0;         m_matrix[3][3] = i_number;

  if(i_number!=1)
  {
  m_inverse_matrix[0][0] = 1. / i_number;   m_inverse_matrix[0][1] = 0;               m_inverse_matrix[0][2] = 0;               m_inverse_matrix[0][3] = 0;
  m_inverse_matrix[1][0] = 0;               m_inverse_matrix[1][1] = 1. / i_number;   m_inverse_matrix[1][2] = 0;               m_inverse_matrix[1][3] = 0;
  m_inverse_matrix[2][0] = 0;               m_inverse_matrix[2][1] = 0;               m_inverse_matrix[2][2] = 1. / i_number;   m_inverse_matrix[2][3] = 0;
  m_inverse_matrix[3][0] = 0;               m_inverse_matrix[3][1] = 0;               m_inverse_matrix[3][2] = 0;               m_inverse_matrix[3][3] = 1. / i_number;
  }
  m_is_identity = false;
  }

TransformationMatrix TransformationMatrix::GetInversed() const
  {
  TransformationMatrix temp(*this);
  temp.Inverse();
  return temp;
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