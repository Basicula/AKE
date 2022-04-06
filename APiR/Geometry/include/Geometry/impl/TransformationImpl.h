#pragma once

template <size_t Dimension>
Transformation<Dimension>::Transformation()
  : m_translation(0.0)
  , m_scale(1.0)
{
  m_rotation.SetIdentity();
  m_inverse_rotation.SetIdentity();
}

template <size_t Dimension>
Transformation<Dimension>::Transformation(const Transformation<Dimension>& i_other)
  : m_translation(i_other.m_translation)
  , m_scale(i_other.m_scale)
  , m_rotation(i_other.m_rotation)
  , m_inverse_rotation(i_other.m_inverse_rotation)
{}

template <size_t Dimension>
Transformation<Dimension> Transformation<Dimension>::GetInversed() const
{
  Transformation temp(*this);
  temp.Inverse();
  return temp;
}

template <size_t Dimension>
typename Transformation<Dimension>::VectorType Transformation<Dimension>::Transform(const VectorType& i_vector,
                                                                                    const bool i_is_vector) const
{
  VectorType vector = i_vector;
  Transform(vector, i_is_vector);
  return vector;
}

template <size_t Dimension>
void Transformation<Dimension>::Transform(VectorType& io_vector, const bool i_is_vector) const
{
  if (i_is_vector)
    m_rotation.ApplyLeft(io_vector); // rotation
  else {
    io_vector *= m_scale;            // scaling
    m_rotation.ApplyLeft(io_vector); // rotation
    io_vector += m_translation;      // translation
  }
}

template <size_t Dimension>
typename Transformation<Dimension>::VectorType Transformation<Dimension>::InverseTransform(const VectorType& i_vector,
                                                                                           const bool i_is_vector) const
{
  VectorType vector = i_vector;
  InverseTransform(vector, i_is_vector);
  return vector;
}

template <size_t Dimension>
void Transformation<Dimension>::InverseTransform(VectorType& io_vector, const bool i_is_vector) const
{
  if (i_is_vector)
    m_inverse_rotation.ApplyLeft(io_vector); // rotation
  else {
    io_vector -= m_translation;              // translation
    m_inverse_rotation.ApplyLeft(io_vector); // rotation
    io_vector /= m_scale;                    // scaling
  }
}

template <size_t Dimension>
typename Transformation<Dimension>::VectorType Transformation<Dimension>::Rotate(const VectorType& i_vector) const
{
  VectorType result = i_vector;
  m_rotation.ApplyLeft(result);
  return result;
}

template <size_t Dimension>
void Transformation<Dimension>::Rotate(VectorType& io_vector) const
{
  m_rotation.ApplyLeft(io_vector);
}

template <size_t Dimension>
typename Transformation<Dimension>::VectorType Transformation<Dimension>::Translate(const VectorType& i_vector) const
{
  return i_vector + m_translation;
}

template <size_t Dimension>
void Transformation<Dimension>::Translate(VectorType& io_vector) const
{
  io_vector += m_translation;
}

template <size_t Dimension>
typename Transformation<Dimension>::VectorType Transformation<Dimension>::Scale(const VectorType& i_vector) const
{
  return i_vector * m_scale;
}

template <size_t Dimension>
void Transformation<Dimension>::Scale(VectorType& io_vector) const
{
  io_vector *= m_scale;
}

template <size_t Dimension>
typename Transformation<Dimension>::VectorType Transformation<Dimension>::GetScale() const
{
  return m_scale;
}

template <size_t Dimension>
void Transformation<Dimension>::SetScale(const VectorType& i_scale)
{
  m_scale = i_scale;
}

template <size_t Dimension>
typename Transformation<Dimension>::VectorType Transformation<Dimension>::GetTranslation() const
{
  return m_translation;
}

template <size_t Dimension>
void Transformation<Dimension>::SetTranslation(const VectorType& i_translation)
{
  m_translation = i_translation;
}

template <size_t Dimension>
typename Transformation<Dimension>::MatrixType Transformation<Dimension>::GetRotation() const
{
  return m_rotation;
}

template <size_t Dimension>
void Transformation<Dimension>::SetRotation(const MatrixType& i_rotation_matrix)
{
  m_rotation = i_rotation_matrix;
  m_inverse_rotation = i_rotation_matrix.Transposed();
}

template <size_t Dimension>
void Transformation<Dimension>::Invert()
{
  m_translation = -m_translation;
  m_scale.Invert();
  std::swap(m_rotation, m_inverse_rotation);
}