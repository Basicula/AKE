#include <Geometry/Transformation.h>

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
  Vector3d local_point = i_world_point;
  local_point -= m_translation; // translation
  m_inverse_rotation.ApplyLeft(local_point); // rotation
  local_point /= m_scale; // scaling
  return local_point;
  }

Vector3d Transformation::PointToWorld(const Vector3d& i_local_point) const
  {
  Vector3d world_point = i_local_point;
  world_point *= m_scale; // scaling
  m_rotation.ApplyLeft(world_point); // rotation
  world_point += m_translation; // translation
  return world_point;
  }

Vector3d Transformation::DirectionToLocal(const Vector3d& i_world_dir) const
  {
  Vector3d local_direction = i_world_dir;
  m_inverse_rotation.ApplyLeft(local_direction); // rotation
  local_direction /= m_scale; // scaling
  local_direction.Normalize(); // keep normalized after scaling
  return local_direction;
  }

Vector3d Transformation::DirectionToWorld(const Vector3d& i_local_dir) const
  {
  Vector3d world_direction = i_local_dir;
  world_direction *= m_scale; // scaling
  m_rotation.ApplyLeft(world_direction); // rotation
  world_direction.Normalize(); // keep normalized after scaling
  return world_direction;
  }
