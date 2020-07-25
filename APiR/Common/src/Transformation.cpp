#include <Transformation.h>

Transformation::Transformation()
  : m_translation(0.0)
  , m_scale(1.0)
  , m_rotation()
  , m_inverse_rotation()
  {}

Transformation::Transformation(const Transformation& i_other)
  : m_translation(i_other.m_translation)
  , m_scale(i_other.m_scale)
  , m_rotation(i_other.m_rotation)
  , m_inverse_rotation(i_other.m_inverse_rotation)
  {}

void Transformation::SetRotation(
  const Vector3d& i_axis,
  double i_degree_in_rad)
  {
  const double cosine = cos(i_degree_in_rad);
  const double one_minus_cos = 1 - cosine;
  const double sine = sin(i_degree_in_rad);
  const double ux = i_axis[0];
  const double uy = i_axis[1];
  const double uz = i_axis[2];
  m_rotation = Matrix3d
    {
    cosine + ux * ux * one_minus_cos,     ux * uy * one_minus_cos - uz * sine,  ux * uz * one_minus_cos + uy * sine,
    uy * ux * one_minus_cos + uz * sine,  cosine + uy * uy * one_minus_cos,     uy * uz * one_minus_cos - ux * sine,
    uz * ux * one_minus_cos - uy * sine,  uz * uy * one_minus_cos + ux * sine,  cosine + uz * uz * one_minus_cos
    };
  m_inverse_rotation = m_rotation.Transposed();
  }

Transformation Transformation::GetInversed() const
  {
  Transformation temp(*this);
  temp.Inverse();
  return temp;
  }

Vector3d Transformation::PointToLocal(const Vector3d& i_world_point) const
  {
  const auto translated = i_world_point - m_translation;
  const auto rotated = m_inverse_rotation * translated;
  const auto scaled = Vector3d(rotated[0] / m_scale[0], rotated[1] / m_scale[1], rotated[2] / m_scale[2]);
  return scaled;
  }

Vector3d Transformation::PointToWorld(const Vector3d& i_local_point) const
  {
  const auto scaled = Vector3d(i_local_point[0] * m_scale[0], i_local_point[1] * m_scale[1], i_local_point[2] * m_scale[2]);
  const auto rotated = m_rotation * scaled;
  const auto translated = rotated + m_translation;
  return translated;
  }

Vector3d Transformation::DirectionToLocal(const Vector3d& i_world_dir) const
  {
  const auto rotated = m_inverse_rotation * i_world_dir;
  const auto scaled = Vector3d(rotated[0] / m_scale[0], rotated[1] / m_scale[1], rotated[2] / m_scale[2]);
  return scaled.Normalized();
  }

Vector3d Transformation::DirectionToWorld(const Vector3d& i_local_dir) const
  {
  const auto scaled = Vector3d(i_local_dir[0] * m_scale[0], i_local_dir[1] * m_scale[1], i_local_dir[2] * m_scale[2]);
  const auto rotated = m_rotation * scaled;
  return rotated.Normalized();
  }
