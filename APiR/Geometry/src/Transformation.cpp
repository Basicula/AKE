#include <Geometry/Transformation.h>

Transformation::Transformation()
  : m_translation(0.0)
  , m_scale(1.0)
  , m_rotation()
  , m_inverse_rotation()
  {
  m_rotation.SetIdentity();
  m_inverse_rotation.SetIdentity();
  }

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
  m_rotation = Matrix3x3d
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

Vector3d Transformation::Transform(const Vector3d& i_vector, const bool i_is_vector) const
  {
  Vector3d vector = i_vector;
  Transform(vector, i_is_vector);
  return vector;
  }

void Transformation::Transform(Vector3d& io_vector, const bool i_is_vector) const
{
  if (i_is_vector)
    m_rotation.ApplyLeft(io_vector); // rotation
  else {
    io_vector *= m_scale; // scaling
    m_rotation.ApplyLeft(io_vector); // rotation
    io_vector += m_translation; // translation
  }
}

Vector3d Transformation::InverseTransform(const Vector3d& i_vector, const bool i_is_vector) const {
  Vector3d vector = i_vector;
  InverseTransform(vector, i_is_vector);
  return vector;
}

void Transformation::InverseTransform(Vector3d& io_vector, const bool i_is_vector) const {
  if (i_is_vector)
    m_inverse_rotation.ApplyLeft(io_vector); // rotation
  else {
    io_vector -= m_translation; // translation
    m_inverse_rotation.ApplyLeft(io_vector); // rotation
    io_vector /= m_scale; // scaling
  }
}

Vector3d Transformation::Rotate(const Vector3d& i_vector) const {
  Vector3d result = i_vector;
  m_rotation.ApplyLeft(result);
  return result;
}

void Transformation::Rotate(Vector3d& io_vector) const {
  m_rotation.ApplyLeft(io_vector);
}

Vector3d Transformation::Translate(const Vector3d& i_vector) const {
  return i_vector + m_translation;
}

void Transformation::Translate(Vector3d& io_vector) const {
  io_vector += m_translation;
}

Vector3d Transformation::Scale(const Vector3d& i_vector) const {
  return i_vector * m_scale;
}

void Transformation::Scale(Vector3d& io_vector) const {
  io_vector *= m_scale;
}
