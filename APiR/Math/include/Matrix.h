#pragma once
#include <type_traits>

//Matrix NxM
/*N = 2, M = 3
| a b c |
| d e f |
*/
//N,M > 1
template<class ElementType, size_t N, size_t M, 
  typename std::enable_if<(N>1)&&(M>1),ElementType>::type* = nullptr>
class Matrix
  {
  public:
    // identity matrix
    Matrix();

    void SetIdentity();

    ElementType operator()(std::size_t i, std::size_t j) const;

  private:
    ElementType m_matrix[N * M];
  };