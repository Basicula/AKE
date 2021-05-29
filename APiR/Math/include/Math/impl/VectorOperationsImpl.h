namespace Math {
  template<typename T, std::size_t Dimension, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
  Vector<T, Dimension> Reflected(const Vector<T, Dimension>& i_normal, const Vector<T, Dimension>& i_vector_to_reflect) {
    return i_vector_to_reflect - i_normal * i_normal.Dot(i_vector_to_reflect) * 2.0;
  }
}