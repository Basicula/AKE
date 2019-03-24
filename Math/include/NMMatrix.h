#pragma once
#include <type_traits>

//Matrix NxM
/*N = 2, M = 3
| a b c |
| d e f |
*/
//N,M > 1
template<class ElementsType, size_t N, size_t M = N, typename std::enable_if<(N>1)&&(M>1),ElementsType>::type* = nullptr>
class NMMatrix
  {
  public:
    NMMatrix();

    void SetIdentity();

  private:
    ElementsType m_matrix[N][M];
  };