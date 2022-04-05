#pragma once

template <>
template <>
inline void Transformation3D::SetRotation(const Vector3d& i_axis, const double i_degree_in_rad)
{
  const double cosine = cos(i_degree_in_rad);
  const double one_minus_cos = 1 - cosine;
  const double sine = sin(i_degree_in_rad);
  const double ux = i_axis[0];
  const double uy = i_axis[1];
  const double uz = i_axis[2];
  m_rotation = Matrix3x3d{ cosine + ux * ux * one_minus_cos,    ux * uy * one_minus_cos - uz * sine,
                           ux * uz * one_minus_cos + uy * sine, uy * ux * one_minus_cos + uz * sine,
                           cosine + uy * uy * one_minus_cos,    uy * uz * one_minus_cos - ux * sine,
                           uz * ux * one_minus_cos - uy * sine, uz * uy * one_minus_cos + ux * sine,
                           cosine + uz * uz * one_minus_cos };
  m_inverse_rotation = m_rotation.Transposed();
}
