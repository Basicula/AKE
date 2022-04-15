#pragma once

template <>
template <>
inline void Transformation2D::SetRotation(const double i_degree_in_rad)
{
  const double cosine = cos(i_degree_in_rad);
  const double sine = sin(i_degree_in_rad);
  m_rotation = Matrix2x2d{ cosine, -sine, sine, cosine };
  m_inverse_rotation = m_rotation.Transposed();
}

template <>
template <>
inline void Transformation2D::Rotate(const double i_degree_in_rad)
{
  const double cosine = cos(i_degree_in_rad);
  const double sine = sin(i_degree_in_rad);
  m_rotation *= Matrix2x2d{ cosine, -sine, sine, cosine };
  m_inverse_rotation = m_rotation.Transposed();
}
