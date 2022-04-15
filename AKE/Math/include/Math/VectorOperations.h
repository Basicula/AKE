#pragma once
#include "Math/Vector.h"

namespace Math {
  template <typename T, std::size_t Dimension, typename std::enable_if<std::is_floating_point<T>::value>::type>
  Vector<T, Dimension> Reflected(const Vector<T, Dimension>& i_normal, const Vector<T, Dimension>& i_vector_to_reflect);
}

#include "impl/VectorOperationsImpl.h"