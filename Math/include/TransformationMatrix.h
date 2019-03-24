#pragma once

#include <Vector.h>

template<class ElementsType = double>
class TransformationMatrix
  {
  public:
    TransformationMatrix();

    template<class ValueType>
    void SetDiagonal(ValueType i_number);
    void SetIdentity();

    inline bool IsIdentity() const { return m_is_identity; };

    inline Vector<3, ElementsType> GetTranslation() const { return Vector<3, ElementsType>(m_matrix[0][3], m_matrix[1][3], m_matrix[2][3]); };

  private:
    bool m_is_identity;
    ElementsType m_matrix[4][4];
  };