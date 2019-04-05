#pragma once

#include <Vector.h>

class TransformationMatrix
  {
  public:
    TransformationMatrix();
    TransformationMatrix(double* i_from16);
    TransformationMatrix(const TransformationMatrix& i_other);

    void SetDiagonal(double i_number);
    void SetIdentity();

    inline bool IsIdentity() const { return m_is_identity; };

    inline Vector3d GetTranslation() const { return Vector3d(m_matrix[0][3], m_matrix[1][3], m_matrix[2][3]); };

    TransformationMatrix operator*(const TransformationMatrix& i_other) const;

  private:
    void _Multiply(const TransformationMatrix& i_other);
    bool _CheckIdentity() const;

  private:
    bool m_is_identity;
    double m_matrix[4][4];
    double m_inverse_matrix[4][4];
  };